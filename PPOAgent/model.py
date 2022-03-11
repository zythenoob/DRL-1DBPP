import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

value_padding = -1e5


# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   log: bool = False,
                   epsilon: float = 1e-20) -> torch.Tensor:
    """
	``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
	masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
	``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
	``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
	broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
	unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
	do it yourself before passing the mask into this function.
	In the case that the input vector is completely masked, the return value of this function is
	arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
	of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
	that we deal with this case relies on having single-precision floats; mixing half-precision
	floats with fully-masked vectors will likely give you ``nans``.
	If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
	lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + epsilon).log()
    if log:
        return torch.log_softmax(vector, dim=dim)
    else:
        return torch.softmax(vector, dim=dim)


# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_max(vector: torch.Tensor,
               mask: torch.Tensor,
               dim: int,
               keepdim: bool = False,
               min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
    """
    To calculate max along certain dimensions on masked values
    Parameters
    ----------
	vector : ``torch.Tensor``
		The vector to calculate max, assume unmasked parts are already zeros
	mask : ``torch.Tensor``
		The mask of the vector. It must be broadcastable with vector.
	dim : ``int``
		The dimension to calculate max
	keepdim : ``bool``
		Whether to keep dimension
	min_val : ``float``
		The minimal value for paddings
	Returns
	-------
	A ``torch.Tensor`` of including the maximum values.
	"""
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index


def pad_tensor(input: torch.Tensor,
               size: int) -> torch.Tensor:
    in_shape = input.shape
    if len(in_shape) == 2:
        output = torch.zeros(in_shape[0], in_shape[1] + size)
        output[:, :in_shape[1]] = input
    elif len(in_shape) == 3:
        output = torch.zeros(in_shape[0], in_shape[1] + size, in_shape[2])
        output[:, :in_shape[1], :] = input
    else:
        raise ValueError('wrong shape')
    return output


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(Encoder, self).__init__()

        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, embedded_inputs, input_lengths):
        self.rnn.flatten_parameters()
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths,
                                                   batch_first=self.batch_first,
                                                   enforce_sorted=False)
        # Forward pass through RNN
        outputs, hidden = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        # Return output and final hidden state
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vp = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        policy = self.vp(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log-softmax for a better numerical stability
        log_policy = masked_softmax(policy, mask, dim=-1, log=True)

        return log_policy, policy


class PointerNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, bidirectional=True, batch_first=True):
        super(PointerNet, self).__init__()

        # Embedding dimension
        self.embedding_dim = embedding_dim
        # (Decoder) hidden size
        self.hidden_size = hidden_size
        # Bidirectional Encoder
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = 1
        self.batch_first = batch_first

        self.embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim, bias=False)
        self.encoder = Encoder(embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=self.num_layers,
                               bidirectional=bidirectional, batch_first=batch_first)
        self.decoding_rnn = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.attn = Attention(hidden_size=hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_seq, input_lengths):

        if self.batch_first:
            batch_size = input_seq.size(0)
            max_seq_len = input_seq.size(1)
        else:
            batch_size = input_seq.size(1)
            max_seq_len = input_seq.size(0)

        # Embedding
        embedded = self.embedding(input_seq)
        # (batch_size, max_seq_len, embedding_dim)

        # encoder_output => (batch_size, max_seq_len, hidden_size) if batch_first else (max_seq_len, batch_size,
        # hidden_size) hidden_size is usually set same as embedding size encoder_hidden => (num_layers *
        # num_directions, batch_size, hidden_size) for each of h_n and c_n
        encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)

        if self.bidirectional:
            # Optionally, Sum bidirectional RNN outputs
            encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]

        encoder_h_n, encoder_c_n = encoder_hidden
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        # Lets use zeros as an intial input for sorting example
        decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        decoder_hidden = (encoder_h_n[-1, 0, :, :], encoder_c_n[-1, 0, :, :])

        range_tensor = torch.arange(max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype).expand(
            batch_size, max_seq_len, max_seq_len)
        each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)

        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor

        pointer_scores = []
        pointer_argmaxs = []

        for i in range(max_seq_len):
            # We will simply mask out when calculating attention or max (and loss later)
            # not all input and hiddens, just for simplicity
            sub_mask = mask_tensor[:, i, :].float().to(torch.device('cuda'))

            # h, c: (batch_size, hidden_size)
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)

            # next hidden
            decoder_hidden = (h_i, c_i)

            # Get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            log_pointer_score, score = self.attn(h_i, encoder_outputs, sub_mask)

            pointer_scores.append(score)

            # Get the indices of maximum pointer
            _, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)

            pointer_argmaxs.append(masked_argmax)
            index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)

            # (batch_size, hidden_size)
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

        # Average to reduce dim 1
        pointer_scores = torch.stack(pointer_scores, 1).sum(dim=1)
        pointer_scores = pointer_scores / pointer_scores.shape[1]

        return pointer_scores


class SelfAttention(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_length, max_seq_length):
        super(SelfAttention, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM

        --------

        """

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length

        self.encoder = nn.LSTM(embedding_length, hidden_size, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size)

        self.W_s1 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.W_s2 = nn.Linear(2 * hidden_size, max_seq_length)

        self.attn_combine = nn.Linear(2 * hidden_size + embedding_length, hidden_size)

        if output_size ==2:
            self.label = nn.Linear(hidden_size, output_size)
        else:
            self.label = nn.Linear(2 * hidden_size * max_seq_length, output_size)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.
        Arguments
        ---------
        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------
        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.
        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)
        """
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_sentences):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        """

        # input.size() = (max_seq_len, batch_size, hidden_size)
        input = input_sentences.permute(1, 0, 2)

        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        output, (h_n, c_n) = self.encoder(input)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        output = output.permute(1, 0, 2)

        # attn_weight_matrix.size() = (batch_size, max_seq_length, num_seq)
        attn_weight_matrix = self.attention_net(output)

        # hidden_matrix.size() = (batch_size, max_seq_length, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)

        if self.output_size ==2:
            # policy
            decoder_hidden = (h_n[0, :, :], c_n[0, :, :])

            output = torch.cat((input[0], hidden_matrix.permute(1, 0, 2)[0]), 1)
            output = torch.tanh(self.attn_combine(output).unsqueeze(0))
            logits, hidden = self.decoder(output, (decoder_hidden[0].unsqueeze(0), decoder_hidden[1].unsqueeze(0)))

            logits = self.label(logits.squeeze(0))
        else:
            # value
            logits = self.label(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
            # logits.size() = (batch_size, output_size)

        return logits
