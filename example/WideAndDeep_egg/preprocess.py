import tarfile
import os
import pickle
import collections
import numpy as np
import tensorflow.compat.v1 as tf
def uncompress(filepath):
    tar = tarfile.open(filepath)
    names = tar.getnames()
    for name in names:
        tar.extract(name,path=os.path.dirname(filepath))
    tar.close()
def mkdir_path(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
class StatsDict():
    """preprocessed data"""

    def __init__(self, field_size, dense_dim, slot_dim, skip_id_convert):
        self.field_size = field_size
        self.dense_dim = dense_dim
        self.slot_dim = slot_dim
        self.skip_id_convert = bool(skip_id_convert)

        self.val_cols = ["val_{}".format(i + 1) for i in range(self.dense_dim)]
        self.cat_cols = ["cat_{}".format(i + 1) for i in range(self.slot_dim)]

        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}

        self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}

        self.oov_prefix = "OOV"

        self.cat2id_dict = {}
        self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
        self.cat2id_dict.update(
            {self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})

    def stats_vals(self, val_list):
        """Handling weights column"""
        assert len(val_list) == len(self.val_cols)

        def map_max_min(i, val):
            key = self.val_cols[i]
            if val != "":
                if float(val) > self.val_max_dict[key]:
                    self.val_max_dict[key] = float(val)
                if float(val) < self.val_min_dict[key]:
                    self.val_min_dict[key] = float(val)

        for i, val in enumerate(val_list):
            map_max_min(i, val)

    def stats_cats(self, cat_list):
        """Handling cats column"""

        assert len(cat_list) == len(self.cat_cols)

        def map_cat_count(i, cat):
            key = self.cat_cols[i]
            self.cat_count_dict[key][cat] += 1

        for i, cat in enumerate(cat_list):
            map_cat_count(i, cat)

    def save_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_max_dict, file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_min_dict, file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.cat_count_dict, file_wrt)

    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_max_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_min_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.cat_count_dict = pickle.load(file_wrt)
        print("val_max_dict.items()[:50]:{}".format(list(self.val_max_dict.items())))
        print("val_min_dict.items()[:50]:{}".format(list(self.val_min_dict.items())))

    def get_cat2id(self, threshold=100):
        for key, cat_count_d in self.cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
            for cat_str, _ in new_cat_count_d.items():
                self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
        print("cat2id_dict.size:{}".format(len(self.cat2id_dict)))
        print("cat2id.dict.items()[:50]:{}".format(list(self.cat2id_dict.items())[:50]))

    def map_cat2id(self, values, cats):
        """Cat to id"""

        def minmax_scale_value(i, val):
            max_v = float(self.val_max_dict["val_{}".format(i + 1)])
            return float(val) * 1.0 / max_v

        id_list = []
        weight_list = []
        for i, val in enumerate(values):
            if val == "":
                id_list.append(i)
                weight_list.append(0)
            else:
                key = "val_{}".format(i + 1)
                id_list.append(self.cat2id_dict[key])
                weight_list.append(minmax_scale_value(i, float(val)))

        for i, cat_str in enumerate(cats):
            key = "cat_{}".format(i + 1) + "_" + cat_str
            if key in self.cat2id_dict:
                if self.skip_id_convert is True:
                    # For the synthetic data, if the generated id is between [0, max_vcoab], but the num examples is l
                    # ess than vocab_size/ slot_nums the id will still be converted to [0, real_vocab], where real_vocab
                    # the actually the vocab size, rather than the max_vocab. So a simple way to alleviate this
                    # problem is skip the id convert, regarding the synthetic data id as the final id.
                    id_list.append(cat_str)
                else:
                    id_list.append(self.cat2id_dict[key])
            else:
                id_list.append(self.cat2id_dict[self.oov_prefix + "cat_{}".format(i + 1)])
            weight_list.append(1.0)
        return id_list, weight_list

def random_split_trans2tfrecord(input_file_path, output_file_path, criteo_stats_dict, part_rows=2000000,
                                  line_per_sample=1000, train_line_count=None,
                                  test_size=0.1, seed=2020, dense_dim=13, slot_dim=26):
    """Random split data and save mindrecord"""
    if train_line_count is None:
        raise ValueError("Please provide training file line count")
    test_size = int(train_line_count * test_size)
    all_indices = [i for i in range(train_line_count)]
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    print("all_indices.size:{}".format(len(all_indices)))
    test_indices_set = set(all_indices[:test_size])
    print("test_indices_set.size:{}".format(len(test_indices_set)))
    print("-----------------------" * 10 + "\n" * 2)

    # train_data_list = []
    # test_data_list = []
    # ids_list = []
    # wts_list = []
    # label_list = []

    # writer_train = FileWriter(os.path.join(output_file_path, "train_input_part.mindrecord"), 21)
    # writer_test = FileWriter(os.path.join(output_file_path, "test_input_part.mindrecord"), 3)
    writer_train = tf.compat.v1.python_io.TFRecordWriter(os.path.join(output_file_path,'train.tfrecord'))
    # writer_test = tf.compat.v1.python_io.TFRecordWriter(os.path.join(output_file_path,'test.tfrecord'))
    

    with open(input_file_path, encoding="utf-8") as file_in:
        items_error_size_lineCount = []
        count = 0
        # train_part_number = 0
        # test_part_number = 0
        for i, line in enumerate(file_in):
            count += 1
            if count % 1000000 == 0:
                print("Have handle {}w lines.".format(count // 10000))
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != (1 + dense_dim + slot_dim):
                items_error_size_lineCount.append(i)
                continue
            label = int(items[0])
            values = items[1:1 + dense_dim]
            cats = items[1 + dense_dim:]

            assert len(values) == dense_dim, "values.size: {}".format(len(values))
            assert len(cats) == slot_dim, "cats.size: {}".format(len(cats))

            ids, wts = criteo_stats_dict.map_cat2id(values, cats)

            # ids_list.extend(ids)
            # wts_list.extend(wts)
            # label_list.append(label)
            ids = np.array(ids,dtype=np.int32)
            ids = ids.tostring()
            wts = np.array(wts,dtype=np.float32)
            wts = wts.tostring()
            if count % 1000000 == 0:
                print("label: ",label)
            example = tf.train.Example(features=tf.train.Features(
                                                                    feature={
                                                                    # Int64List储存int数据
                                                                    'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[label])), 
                                                                    # 储存byte二进制数据
                                                                    'ids':tf.train.Feature(bytes_list = tf.train.BytesList(value=[ids])),
                                                                    'wts':tf.train.Feature(bytes_list = tf.train.BytesList(value=[wts])),
                                                                    }))
            writer_train.write(example.SerializeToString()) 
        writer_train.close()
            # break
        #     if count % line_per_sample == 0:
        #         if i not in test_indices_set:
        #             train_data_list.append({"feat_ids": np.array(ids_list, dtype=np.int32),
        #                                     "feat_vals": np.array(wts_list, dtype=np.float32),
        #                                     "label": np.array(label_list, dtype=np.float32)
        #                                     })
        #         else:
        #             test_data_list.append({"feat_ids": np.array(ids_list, dtype=np.int32),
        #                                    "feat_vals": np.array(wts_list, dtype=np.float32),
        #                                    "label": np.array(label_list, dtype=np.float32)
        #                                    })
        #         if train_data_list and len(train_data_list) % part_rows == 0:
        #             # writer_train.write_raw_data(train_data_list)
        #             train_data_list.clear()
        #             train_part_number += 1

        #         if test_data_list and len(test_data_list) % part_rows == 0:
        #             # writer_test.write_raw_data(test_data_list)
        #             test_data_list.clear()
        #             test_part_number += 1

        #         ids_list.clear()
        #         wts_list.clear()
        #         label_list.clear()

        # if train_data_list:
        #     # writer_train.write_raw_data(train_data_list)
        # if test_data_list:
        #     # writer_test.write_raw_data(test_data_list)
    

    print("-------------" * 10)
    print("items_error_size_lineCount.size(): {}.".format(len(items_error_size_lineCount)))
    print("-------------" * 10)
    np.save("items_error_size_lineCount.npy", items_error_size_lineCount)
def statsdata(file_path, dict_output_path, criteo_stats_dict, dense_dim=13, slot_dim=26):
    """Preprocess data and save data"""
    with open(file_path, encoding="utf-8") as file_in:
        errorline_list = []
        count = 0
        for line in file_in:
            count += 1
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != (dense_dim + slot_dim + 1):
                errorline_list.append(count)
                print("Found line length: {}, suppose to be {}, the line is {}".format(len(items),
                                                                                       dense_dim + slot_dim + 1, line))
                continue
            if count % 1000000 == 0:
                print("Have handled {}w lines.".format(count // 10000))
            values = items[1: dense_dim + 1]
            cats = items[dense_dim + 1:]

            assert len(values) == dense_dim, "values.size: {}".format(len(values))
            assert len(cats) == slot_dim, "cats.size: {}".format(len(cats))
            criteo_stats_dict.stats_vals(values)
            criteo_stats_dict.stats_cats(cats)
    criteo_stats_dict.save_dict(dict_output_path)
if __name__ == "__main__":
    
    target_field_size = 39
    dense_dim=13
    slot_dim=26
    threshold=100
    skip_id_convert=0
    train_line_count=45840617
    data_file_path = '/home/haiqwa/dataset/criteo/origin/train.txt'
    stats_output_path = '/home/haiqwa/dataset/criteo/stats_dict/'
    output_path = '/home/haiqwa/dataset/criteo/tfrecord/'
    mkdir_path(stats_output_path)
    mkdir_path(output_path)
    stats = StatsDict(field_size=target_field_size, dense_dim=dense_dim, slot_dim=slot_dim,
                      skip_id_convert=skip_id_convert)
    statsdata(data_file_path, stats_output_path, stats, dense_dim=dense_dim, slot_dim=slot_dim)
    stats.load_dict(dict_path=stats_output_path, prefix="")
    stats.get_cat2id(threshold=threshold)
    random_split_trans2tfrecord(data_file_path, output_path, stats, part_rows=2000000,
                                  train_line_count=train_line_count, line_per_sample=1000,
                                  test_size=0.1, seed=2020, dense_dim=dense_dim, slot_dim=slot_dim)