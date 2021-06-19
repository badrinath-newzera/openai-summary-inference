import functools
import os
from summarize_from_feedback.sample import main, HParams
from summarize_from_feedback.query_response_model import ModelSpec, SampleHParams, RunParams
from summarize_from_feedback.tasks import TaskHParams, TaskQueryHParams, TaskResponseHParams


print("Hi")
# SUBREDDIT: r/{subreddit}\n\n
# took a really difficult math class, didn't get it the night before the final, woke up the next morning and aced it and now go drinking the night before every final.  What are your final routines?
# TITLE: {title}\n\nPOST: {post}\n\nTL;DR:
# Article:\n\n{article}\n\nTL;DR:
call = functools.partial(main, HParams(model_spec=ModelSpec(device='cuda', load_path='https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/sup4_ppo_rm4', use_cache=True, short_name='sup4_ppo_rm4', init_heads=None, map_heads={}, run_params=RunParams(fp16_embedding_weights=False, fp16_conv_weights=False, attn_dropout=0.0, resid_dropout=0.0, emb_dropout=0.0, n_shards=1)), orig_model_spec=None, task=TaskHParams(query=TaskQueryHParams(length=512, dataset='tldr_3_filtered',
                         format_str='Article:\n\n{article}\n\nTL;DR:', truncate_field='article', truncate_text='\n', padding=None, pad_side='left'), response=TaskResponseHParams(ref_format_str=' {reference}', length=48, truncate_token=50256)), query_dataset_split='valid', sample=SampleHParams(temperature=0.01, top_p=1.0), num_queries=736, queries_per_run_per_replica=1, responses_per_query=1, responses_per_query_per_batch=1, seed=0, fp16_activations=True))

env = os.environ.copy()
env["JOB_NAME"] = "sample-ppo-xl"

call()
