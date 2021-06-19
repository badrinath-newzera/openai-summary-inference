import json

from summarize_from_feedback.utils import blobs

# ['id', 'subreddit', 'title', 'post', 'summary']
# t3_1ov8e0 -> id
# and k != "subreddit"


def tldr_filtered_generator(split):
    assert split in ["test", "train", "valid"]

    f = open('results_new.json', 'r')
    datas = json.load(f)
    datas = json.loads(datas)
    datas = datas['results']
    for data in datas:
        for artcle in data['articles']:
            yield dict(reference=data["event_synopis"], article=artcle['headline'] + ' ' + " ".join(artcle['article_body'].split()[:100]))


def tldr_filtered_queries_generator(split):
    assert split in ["test", "train", "valid"]

    gcs_path = f"https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered_queries/{split}.jsonl"
    with blobs.open_file_cached(gcs_path, "rb") as f:
        datas = [json.loads(l) for l in f.readlines()]

    for data in datas:
        # NOTE: don't use ref summary, not filtered
        yield dict(reference=data["summary"], **{k: v for (k, v) in data.items() if k != "summary"})


if __name__ == "__main__":
    for x in tldr_filtered_generator("train"):
        print(list(x.keys()))
        break
