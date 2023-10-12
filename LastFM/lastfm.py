import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split


max_time_delta = pd.Timedelta(minutes=15)
min_trail_time = pd.Timedelta(minutes=1)
# top_n_songs = 100
# n_rows = 10000


def extract_trails(max_time_delta, min_trail_time, top_n_songs, n_rows):
    rows = pd.read_csv(
        'LastFM/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv',
        sep='\t',
        nrows=n_rows,
        header=None,
        names=["userid", "timestamp", "artid", "artname", "traid", "traname"],
        on_bad_lines='skip')
    rows.timestamp = pd.to_datetime(rows.timestamp)

    trails = []

    groups = rows.groupby("userid")
    i = 0
    for _, g in groups:
        g.sort_values("timestamp", inplace=True)
        prev_timestamp = pd.Timestamp(0, tz="UTC")
        trail = []
        trail_time = pd.Timedelta(0)
        for row in g.itertuples():
            timestamp = row.timestamp
            traname = row.traname
            time_delta = timestamp - prev_timestamp
            trail_time += time_delta

            if time_delta > max_time_delta:
                prev_timestamp = timestamp
                if trail_time > min_trail_time: trails.append((i, trail))
                # print(time_delta, trail)
                trail = []
            
            else:
                trail.append((traname, int(time_delta.total_seconds())))

        if trail_time >= min_trail_time: trails.append((i, trail))

        if i % 10 == 0:
            print(f"{i} / {len(groups)} users processed: {100 * i / len(groups):.2f}%")
        i += 1


    song_counts = {}

    for label, trail in trails:
        for song, _ in trail:
            if song not in song_counts:
                song_counts[song] = 0
            song_counts[song] += 1

    popular_songs = pd.Series(song_counts).nlargest(top_n_songs).index
    song_ids = { song: i for song, i in zip(popular_songs, range(top_n_songs))}


    trails_ = []
    labels_ = []

    for label, trail in trails:
        trail_ = []
        trail_time = 0
        for song, t in trail:
            if song in song_ids:
                trail_.append((song_ids[song], t))
                trail_time += t
        if trail_time >= min_trail_time.total_seconds():
            labels_.append(label)
            trails_.append(trail_)

    return trails_, labels_


def filename_ct(max_time_delta, min_trail_time, top_n_songs, n_rows):
    return f"trails-{max_time_delta.total_seconds():.0f}-{min_trail_time.total_seconds():.0f}-{top_n_songs}-{n_rows or 'all'}.h5"

def load_ct(max_time_delta, min_trail_time, top_n_songs, n_rows, seed=None):
    filename = filename_ct(max_time_delta, min_trail_time, top_n_songs, n_rows)

    try:
        trails_ct = list(pd.read_hdf(filename, "trails"))
        labels_ct = list(pd.read_hdf(filename, "labels"))

    except (OSError, IOError) as e:
        trails_ct, labels_ct = extract_trails(max_time_delta, min_trail_time, top_n_songs, n_rows)
        pd.Series(trails_ct).to_hdf(filename, "trails")
        pd.Series(labels_ct).to_hdf(filename, "labels")

    trails_ct_train, trails_ct_test, labels_ct_train, labels_ct_test = train_test_split(trails_ct, labels_ct, test_size=0.2, random_state=seed)
    ct_train = (trails_ct_train, labels_ct_train)
    ct_test = (trails_ct_test, labels_ct_test)
    return ct_train, ct_test


def filename_dt(max_time_delta, min_trail_time, top_n_songs, n_rows, tau, t_len):
    return f"dt-{tau}-{t_len}-{filename_ct(max_time_delta, min_trail_time, top_n_songs, n_rows)}"

def load_dt(max_time_delta, min_trail_time, top_n_songs, n_rows, tau, t_len=None):
    hf = h5py.File(filename_dt(max_time_delta, min_trail_time, top_n_songs, n_rows, tau, t_len), 'a')
    if "trails-train" in hf.keys():
        dt_trails_train = hf.get("trails-train")
        dt_labels_train = hf.get("labels-train")
        dt_trails_test = hf.get("trails-test")
        dt_labels_test = hf.get("labels-test")

    else:
        ct_train, ct_test = load_ct(
            max_time_delta=max_time_delta,
            min_trail_time=min_trail_time,
            top_n_songs=top_n_songs,
            n_rows=n_rows)
        ct_trails_train, ct_labels_train = ct_train
        ct_trails_test, ct_labels_test = ct_test

        dt_trails_train = []
        dt_labels_train = []
        dt_trails_test = []
        dt_labels_test = []
        for ct_trails, ct_labels, dt_trails, dt_labels in [(ct_trails_train, ct_labels_train, dt_trails_train, dt_labels_train), (ct_trails_test, ct_labels_test, dt_trails_test, dt_labels_test)]:
            for label, ct_trail in zip(ct_labels, ct_trails):
                dt_trail = []
                ct_time = 0
                for x, t in ct_trail:
                    ct_time += t
                    while ct_time >= tau:
                        dt_trail.append(x)
                        ct_time -= tau
                        if t_len is not None and len(dt_trail) >= t_len:
                            dt_trails.append(dt_trail)
                            dt_labels.append(label)
                            dt_trail = []
                if t_len is None:
                    dt_trails.append(dt_trail)
                    dt_labels.append(label)
        
        hf.create_dataset("trails-train", data=dt_trails_train)
        hf.create_dataset("labels-train", data=dt_labels_train)
        hf.create_dataset("trails-test", data=dt_trails_test)
        hf.create_dataset("labels-test", data=dt_labels_test)

    dt_train = (np.array(dt_trails_train), np.array(dt_labels_train))
    dt_test = (np.array(dt_trails_test), np.array(dt_labels_test))
    return dt_train, dt_test



def text_classification_learn(data):
    import torch
    from torch.utils.data import Dataset, DataLoader

    train_iter, test_iter = data(split="train"), data(split="test")


    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield text

    text_pipeline = lambda x: x
    label_pipeline = lambda x: x


    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)


    dataloader = DataLoader(
        train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch
    )


    from torch import nn

    class TextClassificationModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_class):
            super(TextClassificationModel, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
            self.fc = nn.Linear(embed_dim, num_class)
            self.init_weights()

        def init_weights(self):
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()

        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            return self.fc(embedded)


    num_class = 2
    vocab_size = 14
    emsize = 1024 # 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)


    import time

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(
                        epoch, idx, len(dataloader), total_acc / total_count
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time.time()


    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count


    from torch.utils.data.dataset import random_split
    from torchtext.data.functional import to_map_style_dataset

    # Hyperparameters
    EPOCHS = 20  # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64  # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train]
    )

    train_dataloader = DataLoader(
        split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    valid_dataloader = DataLoader(
        split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)


    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader)
    accu_train = evaluate(train_dataloader)
    print("test accuracy {:8.3f}".format(accu_test))

    return accu_train, accu_test




n_rows = None
#        19150868 <- total number of rows

#        len   total
# user1: 16685 16685
# user2: 57438 74123
# user3: 19494 93617
# user4: 18411 112028
# user5: 20341 132369


# for top_n_songs in [10, 20, 50, 100]:
#     load_ct(max_time_delta=max_time_delta, min_trail_time=min_trail_time, top_n_songs=top_n_songs, n_rows=n_rows)

# trails_ct = load_ct(max_time_delta=max_time_delta, min_trail_time=min_trail_time, top_n_songs=10, n_rows=n_rows)
# extract_trails(max_time_delta=max_time_delta, min_trail_time=min_trail_time, top_n_songs=10, n_rows=n_rows)

# trails_dt, labels = load_dt(max_time_delta=max_time_delta, min_trail_time=min_trail_time, top_n_songs=10, n_rows=n_rows, tau=10, t_len=100)
