#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

using namespace std;

static void print_bin(const vector<double>& arr) {
    cout << "[";
    for (auto & i : arr) {
        cout << i << ", ";
    }
    cout << "]" << endl;
}

static double sum(const vector<double>& arr) {
    double sum = 0;
    for (auto & i : arr) { sum += i; }
    return sum;
}

static void remove_element(vector<double>& arr, double e) {
    arr.erase(arr.begin() + distance(begin(arr), find(arr.begin(), arr.end(), e)));
}

static vector<int> where_equal(vector<int>& arr, int v) {
    vector<int> indexes;
    for (int i=0; i<arr.size(); i++) {
        if (arr[i] == v) { indexes.push_back(i); }
    }
    return indexes;
}

static vector<double> hist2items(const vector<int>& hist, double unit_size) {
    vector<double> items;
    for (int i=0; i<hist.size(); i++) {
        double item = (unit_size * i + unit_size * (i+1)) / 2.0;
        for (int n=0; n<hist[i]; n++) { items.push_back(item); }
    }

    shuffle(items.begin(), items.end(), default_random_engine(0));
    return items;
}

class MBS {
public:
    MBS() {
        srand(0);
    }
    int online_mbs(vector<vector<double>>& bins, vector<double> items, vector<int> histogram, double min_item, int step);
    int mbs_expert_demo(vector<vector<double>>& bins, double cur_item, vector<double> items, double min_item);
    vector<double> relaxed_mbs_predict(vector<double> bin, vector<double> items);
    vector<vector<double>> relaxed_mbs_fit(vector<vector<double>> bins, vector<double> items, double min_item);
    void mbs_one_packing(vector<double>& bin, int index, vector<double>& best_bin);
    bool mbs_pack_history_contains(const vector<double>& bin);
    double mbs_best = 0;
    double relax = 0.99;
    int hist_cats = 25;
    vector<vector<double>> mbs_pack_history;
    vector<double> mbs_items;
};

bool MBS::mbs_pack_history_contains(const vector<double>& bin) {
    for (auto & i : mbs_pack_history) {
        if (i.size() != bin.size()) { continue; }
        bool all_same = true;
        for (int j=0; j < i.size(); j++) {
            if (i[j] != bin[j]) { all_same = false; }
        }
        if (all_same) { return true; }
    }
    return false;
}

void MBS::mbs_one_packing(vector<double>& bin, int index, vector<double>& best_bin) {
    for (int i=index; i < mbs_items.size(); i++) {
//        cout << endl;
//        cout << "current: ";
//        print_bin(bin);
//        cout << "checking: " << mbs_items[i] << endl;
        if (sum(bin) + mbs_items[i] <= 1.0) {
//            cout << "can pack" << endl;
            bin.push_back(mbs_items[i]);
            if (!mbs_pack_history_contains(bin)) {
//                cout << "not in history" << endl;
                mbs_pack_history.push_back(bin);
                mbs_one_packing(bin, i+1, best_bin);
                bin.pop_back();
            } else {
                bin.pop_back();
                continue;
            }
            if (mbs_best > relax) { return; }
        }
    }
    double util = sum(bin);
    if (util - mbs_best > 1e-6 or (abs(util-mbs_best) < 1e-6 and bin.size() < best_bin.size())) {
//        cout << "##### find best: " << util << endl;
        mbs_best = util;
        best_bin.clear();
        best_bin.assign(bin.begin(), bin.end());
    }
}

// initialize function for MBSOnePacking search
vector<double> MBS::relaxed_mbs_predict(vector<double> bin, vector<double> items) {
    vector<double> best_bin;
    mbs_items.assign(items.begin(), items.end());
    mbs_one_packing(bin, 0, best_bin);
    mbs_pack_history.clear();
    mbs_items.clear();
    mbs_best = 0;
    return best_bin;
}

// offline MBS
vector<vector<double>> MBS::relaxed_mbs_fit(vector<vector<double>> bins, vector<double> items, double min_item) {
    sort(items.rbegin(), items.rend());
//    cout << "fit start" << endl;
    if (bins.empty() and !items.empty()) {
        bins.push_back({items[0]});
        items.erase(items.begin());
    }

//    cout << "items: ";
//    print_bin(items);
//    cout << "bins: ";
//    for (auto & b : bins) {print_bin(b);}
//    cout << "fit loop" << endl;

    while (!items.empty()) {
//        cout << endl;
        for (auto & bin : bins) {
            if (sum(bin) <= 1 - min_item) {
//                cout << "fitting bin: " << 0 << ", len items: " << items.size() << endl;
                int bin_len = bin.size();
                vector<double> pred_bin = relaxed_mbs_predict(bin, items);
//                cout << "pred bin:";
//                print_bin(pred_bin);

                vector<double> packed_items;
                auto k = pred_bin.begin() + bin_len;
                auto j = pred_bin.end();
                packed_items.assign(k, j);
                bin.clear();
                bin.assign(pred_bin.begin(), pred_bin.end());

                for (auto & n : packed_items) { remove_element(items, n); }
//                cout << "fitted: ";
//                print_bin(bin);
            }
        }

        if (items.empty()) { break; }

//        cout << "new bin, len items: " << items.size() << endl;
        vector<double> new_bin = {items[0]};
        items.erase(items.begin());
        vector<double> pred_bin = relaxed_mbs_predict(new_bin, items);
        vector<double> packed_items;
        auto k = pred_bin.begin() + 1;
        auto j = pred_bin.end();
        packed_items.assign(k, j);
        for (auto & n : packed_items) { remove_element(items, n); }
        bins.push_back(pred_bin);
//        cout << "fitted: ";
//        print_bin(pred_bin);
    }

//    cout << "pred bins:" << endl;
//    for (auto & b : bins) {print_bin(b);}
    return bins;
}

/*
*  Online MBS Version 1
*  step: only run for 1 step
*/
int MBS::online_mbs(vector<vector<double>>& bins, vector<double> items, vector<int> histogram,
                    double min_item, int step=0) {
    double unit_item_size = 1.0 / (double) hist_cats;
    vector<vector<double>> closed_bins;

    while (!items.empty()) {
        double item = items[0];
        items.erase(items.begin());

//        cout << endl;
//        cout << "item: " << item << endl;
//        cout << "histogram: [";
//        for (auto & h : histogram) { cout << h << ", "; }
//        cout << "]" << endl;
//        for (auto & b : bins) {print_bin(b);}

        if (step == 0) {
            int hist_index = ceil(item * (double)hist_cats) - 1;
            histogram[hist_index] -= 1;
        }

        // generate item sequence by histogram
        vector<double> pred_item_sequence = hist2items(histogram, unit_item_size);

        // 1st level
        vector<int> pred_mbs_bins(bins.size() + 1, 999);
        vector<vector<double>> temp_bins;
        temp_bins.assign(bins.begin(), bins.end());
        vector<double> empty_bin;
        temp_bins.push_back(empty_bin);
        relax = 0.99;

        // predict final bin usages for packing current item into each bin in temp_bins
        for (int i=0; i<temp_bins.size(); i++) {
            if (sum(temp_bins[i]) + item <= 1.0) {
                temp_bins[i].push_back(item);
                pred_mbs_bins[i] = relaxed_mbs_fit(temp_bins, pred_item_sequence, min_item).size();
                temp_bins[i].pop_back();
            }
        }

//        cout << "level 1 pred: ";
//        cout << "[";
//        for (auto & i : pred_mbs_bins) {
//            cout << i << ", ";
//        }
//        cout << "]" << endl;

        // get bin indexes with minimal bin usage
        vector<int> indexes = where_equal(pred_mbs_bins, *(min_element(pred_mbs_bins.begin(), pred_mbs_bins.end())));
        int index = -1;
        // multiple minimum, 2nd level
        if (indexes.size() > 1) {
            vector<double> pred_next;
            for (auto & i : indexes) {
                temp_bins[i].push_back(item);
                bool new_in_temp = false;
                if (i == temp_bins.size() - 1) {
                    vector<double> empty_bin_2;
                    temp_bins.push_back(empty_bin_2);
                    new_in_temp = true;
                }

                vector<double> next_item_pred;
                // get weighted average final bin usage predictions by calculating predictions for each next possible item
                for (int m=0; m<histogram.size(); m++) {
                    if (histogram[m] != 0) {
                        // next item: lies in histogram[m]
                        double item_next = (unit_item_size * m + unit_item_size * (m+1)) / 2.0;
                        histogram[m] -= 1;
                        vector<double> pred_items_sequence_next = hist2items(histogram, unit_item_size);

                        // do the same as 1st level
                        vector<int> pred_mbs_bins_next(temp_bins.size(), 999);
                        relax = 0.99;
                        for (int j=0; j<temp_bins.size(); j++) {
                            if (sum(temp_bins[j]) + item_next <= 1.0) {
                                temp_bins[j].push_back(item_next);
                                pred_mbs_bins_next[j] = relaxed_mbs_fit(temp_bins, pred_items_sequence_next, min_item).size();
                                temp_bins[j].pop_back();
                            }
                        }

                        histogram[m] += 1;
                        next_item_pred.push_back((double)*(min_element(pred_mbs_bins_next.begin(), pred_mbs_bins_next.end())) * (double) histogram[m] / (double)accumulate(histogram.begin(), histogram.end(), 0));
                    }
                }
                pred_next.push_back(sum(next_item_pred));
                temp_bins[i].pop_back();
                if (new_in_temp) {
                    temp_bins.pop_back();
                }
            }
//            cout << "level 2 pred: ";
//            print_bin(pred_next);
            index = indexes[distance(begin(pred_next), find(pred_next.begin(), pred_next.end(), *(min_element(pred_next.begin(), pred_next.end()))))];
        } else {
            index = indexes[0];
        }

//        cout << "action: " << index << endl;
        // pack
        if (index == bins.size()) {
            bins.push_back({item});
            if (step == 1) { return -1; }
        } else {
            bins[index].push_back(item);
            if (step == 1) { return index; }
        }

        // close bin if cannot pack more items
        if (sum(bins[index]) > 1 - min_item) {
            closed_bins.push_back(bins[index]);
            bins.erase(bins.begin() + index);
        }
    }

    bins.insert(bins.end(), closed_bins.begin(), closed_bins.end());
//    for (auto & b : bins) {
//        if (sum(b) > 1.0 or b.size() == 0) {
//            cout << "error: ";
//        }
//        print_bin(b);
//    }
    return bins.size();
}


/*
* Online MBS Version 2
*/
//int MBS::online_mbs(vector<vector<double>>& bins, vector<double> items, vector<int> histogram,
//                    double min_item, int step=0) {
//    double unit_item_size = 1.0 / (double) hist_cats;
//    vector<vector<double>> closed_bins;
//
//    while (!items.empty()) {
//        double item = items[0];
//        items.erase(items.begin());
//
//        if (step == 0) {
//            int hist_index = ceil(item * (double)hist_cats) - 1;
//            histogram[hist_index] -= 1;
//        }
//
//        // generate item sequence by histogram
//        vector<double> pred_item_sequence = { item };
//        for (auto & i : hist2items(histogram, unit_item_size)) {
//            pred_item_sequence.push_back(i);
//        }
//
//        vector<vector<double>> temp_bins;
//        temp_bins.assign(bins.begin(), bins.end());
//        relax = 1.0; // no relax
//
////        cout << endl;
////        cout << "item: " << item << endl;
////        cout << "bins: ";
////        for (auto & b : bins) {
////            print_bin(b);
////        }
//        int bins_size = temp_bins.size();
////        cout << "size: " << bins_size << endl;
//        vector<vector<double>> fitted_bins = relaxed_mbs_fit(temp_bins, pred_item_sequence, min_item);
////        cout << "pred items: ";
////        print_bin(pred_item_sequence);
////        cout << "fitted: " << endl;
////        for (auto & b : fitted_bins) {
////            print_bin(b);
////        }
//
//        int pack_pos;
//        for (pack_pos=0; pack_pos<fitted_bins.size(); pack_pos++) {
//            if (pack_pos == bins_size) {
//                pack_pos = -1;
//                break;
//            }
//            if (count(fitted_bins[pack_pos].begin(), fitted_bins[pack_pos].end(), item) > 0 and
//                count(fitted_bins[pack_pos].begin(), fitted_bins[pack_pos].end(), item) > count(temp_bins[pack_pos].begin(), temp_bins[pack_pos].end(), item)) {
//                break;
//            }
//        }
////        cout << "pack pos: " << pack_pos << endl;
//
//        // pack
//        if (pack_pos == -1) {
//            bins.push_back({item});
//            if (step == 1) { return -1; }
//            pack_pos = bins_size;
//        } else {
//            bins[pack_pos].push_back(item);
//            if (step == 1) { return pack_pos; }
//        }
////        cout << "after pack: " << endl;
////        for (auto & b : bins) {
////            print_bin(b);
////        }
//
//        if (sum(bins[pack_pos]) > 1 - min_item) {
//            closed_bins.push_back(bins[pack_pos]);
//            bins.erase(bins.begin() + pack_pos);
//        }
//    }
//
//    bins.insert(bins.end(), closed_bins.begin(), closed_bins.end());
//    return bins.size();
//}


int best_fit(vector<vector<double>> bins, vector<double> items) {
    for (auto & i : items) {
        double best = 0;
        int action = -1;
        for (int j=0; j<bins.size(); j++) {
            double s = sum(bins[j]) + i;
            if (bins[j].size() > 0 and s > best and s <= 1.0) {
                best = s;
                action = j;
            }
        }
        if (action == -1) {
            vector<double> new_bin = { i };
            bins.push_back(new_bin);
        } else {
            bins[action].push_back(i);
        }
    }
    return bins.size();
}


int best_fit_step(vector<vector<double>>& bins, double item, double thresh=1.0) {
    double best = 0;
    int action = -1;
    for (int j=0; j<bins.size(); j++) {
        double s = sum(bins[j]) + item;
        if (bins[j].size() > 0 and s > best and s <= thresh) {
            best = s;
            action = j;
        }
    }
    if (action == -1) {
        for (int j=0; j<bins.size(); j++) {
            if (bins[j].size() == 0) {
                action = j;
                break;
            }
        }
    }
    return action;
}

int worst_fit_step(vector<vector<double>>& bins, double item, double thresh=1.0) {
    double worst = 0.0;
    int action = -1;
    for (int j=0; j<bins.size(); j++) {
        double s = 1.0 - sum(bins[j]) - item;
        if (bins[j].size() > 0 and s <= thresh and s > worst) {
            worst = s;
            action = j;
        }
    }
    return action;
}

int first_fit_step(vector<vector<double>>& bins, double item) {
    for (int j=0; j<bins.size(); j++) {
        double s = 1.0 - sum(bins[j]) - item;
        if (bins[j].size() > 0 and s >= 0.0) {
            return j;
        }
    }
    return -1;
}

void construct_bins(double in[][50], const int h, const int w, vector<vector<double>>& bins) {
    for (int i=0; i<h; i++) {
        vector<double> b;
        for (int j=0; j<w; j++) {
            if (in[i][j] == 0) { break; }
            b.push_back(in[i][j]);
        }
        bins.push_back(b);
    }
}

//// online mbs expert demo (directly uses item list rather than histogram, pack for 1 step)
//int MBS::mbs_expert_demo(vector<vector<double>>& bins, double cur_item, vector<double> items, double min_item) {
//    vector<vector<double>> closed_bins;
//
//    // 1st level
//    vector<int> pred_mbs_bins(bins.size() + 1, 999);
//    vector<vector<double>> temp_bins;
//    temp_bins.assign(bins.begin(), bins.end());
//    vector<double> empty_bin;
//    temp_bins.push_back(empty_bin);
//    relax = 1.0;
//
//    for (int i=0; i<temp_bins.size(); i++) {
//        if (sum(temp_bins[i]) + cur_item <= 1.0) {
//            temp_bins[i].push_back(cur_item);
//            pred_mbs_bins[i] = relaxed_mbs_fit(temp_bins, items, min_item).size();
//            temp_bins[i].pop_back();
//        }
//    }
//
//    vector<int> mbs_indexes = where_equal(pred_mbs_bins, *(min_element(pred_mbs_bins.begin(), pred_mbs_bins.end())));
//    int index = mbs_indexes[0];
//
//    if (index == bins.size()) {
//        return -1;
//    } else {
//        return index;
//    }
//}

// online mbs expert demo (a different idea than the above one)
int MBS::mbs_expert_demo(vector<vector<double>>& bins, double cur_item, vector<double> items, double min_item) {
    int bins_size = bins.size();
//    cout << "items: ";
//    print_bin(items);
//    cout << "bins:" << endl;
//    for (auto & b : bins) { print_bin(b);}
    vector<vector<double>> fitted_bins = relaxed_mbs_fit(bins, items, min_item);
//    cout << "fitted:" << endl;
//    for (auto & b : fitted_bins) { print_bin(b);}

    int pack_pos;
    for (pack_pos=0; pack_pos<fitted_bins.size(); pack_pos++) {
        if (pack_pos == bins_size or (pack_pos == bins_size - 1 and bins_size == fitted_bins.size())) { return -1; }
        if (count(fitted_bins[pack_pos].begin(), fitted_bins[pack_pos].end(), cur_item) > 0 and
            count(fitted_bins[pack_pos].begin(), fitted_bins[pack_pos].end(), cur_item) > count(bins[pack_pos].begin(), bins[pack_pos].end(), cur_item)) {
            return pack_pos;
        }
    }
}


extern "C" {
    int c_online_mbs(double in[][50], const int h, const int w, double* in_items, const int n_items, int* in_hist, double min_item, int step) {
        auto* mbs = new MBS();

        vector<vector<double>> bins;
        construct_bins(in, h, w, bins);

        vector<double> items;
        for (int i=0; i<n_items; i++) {
            items.push_back(in_items[i]);
        }

        vector<int> hist;
        for (int i=0; i<mbs->hist_cats*2; i += 2) {
            hist.push_back(in_hist[i]);
        }
        return mbs->online_mbs(bins, items, hist, min_item, step);
    }

    int c_offline_mbs(double in[][50], const int h, const int w, double* in_items, const int n_items, double min_item, double relax) {
        auto* mbs = new MBS();

        vector<vector<double>> bins;
        construct_bins(in, h, w, bins);

        vector<double> items;
        for (int i=0; i<n_items; i++) {
            items.push_back(in_items[i]);
        }

        mbs->relax = relax;
        return mbs->relaxed_mbs_fit(bins, items, min_item).size();
    }

    int c_mbs_demo(double in[][50], const int h, const int w, double item, double* in_items, const int n_items, double min_item, double relax) {
        auto* mbs = new MBS();

        vector<vector<double>> bins;
        construct_bins(in, h, w, bins);

        vector<double> items;
        for (int i=0; i<n_items; i++) {
            items.push_back(in_items[i]);
        }

        mbs->relax = relax;
        return mbs->mbs_expert_demo(bins, item, items, min_item);
    }

    int c_bf(double in[][50], const int h, const int w, double* in_items, const int n_items) {
        vector<vector<double>> bins;
        construct_bins(in, h, w, bins);

        vector<double> items;
        for (int i=0; i<n_items; i++) {
            items.push_back(in_items[i]);
        }

        return best_fit(bins, items);
    }

    int c_bf_step(double in[][50], const int h, const int w, double item, double thresh) {
        vector<vector<double>> bins;
        construct_bins(in, h, w, bins);
        return best_fit_step(bins, item, thresh);
    }

    int c_wf_step(double in[][50], const int h, const int w, double item, double thresh) {
        vector<vector<double>> bins;
        construct_bins(in, h, w, bins);
        return worst_fit_step(bins, item, thresh);
    }

    int c_ff_step(double in[][50], const int h, const int w, double item) {
        vector<vector<double>> bins;
        construct_bins(in, h, w, bins);
        return first_fit_step(bins, item);
    }
}
