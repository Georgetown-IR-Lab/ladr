# Instructions for reproducing the SIGIR 2023 paper **Lexically-Accelerated Dense Retrieval**

If you use this code please cite the following paper:

```
@inproceedings{10.1145/3539618.3591715,
author = {Kulkarni, Hrishikesh and MacAvaney, Sean and Goharian, Nazli and Frieder, Ophir},
title = {Lexically-Accelerated Dense Retrieval},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591715},
doi = {10.1145/3539618.3591715},
abstract = {Retrieval approaches that score documents based on learned dense vectors (i.e., dense retrieval) rather than lexical signals (i.e., conventional retrieval) are increasingly popular. Their ability to identify related documents that do not necessarily contain the same terms as those appearing in the user's query (thereby improving recall) is one of their key advantages. However, to actually achieve these gains, dense retrieval approaches typically require an exhaustive search over the document collection, making them considerably more expensive at query-time than conventional lexical approaches. Several techniques aim to reduce this computational overhead by approximating the results of a full dense retriever. Although these approaches reasonably approximate the top results, they suffer in terms of recall -- one of the key advantages of dense retrieval. We introduce 'LADR' (Lexically-Accelerated Dense Retrieval), a simple-yet-effective approach that improves the efficiency of existing dense retrieval models without compromising on retrieval effectiveness. LADR uses lexical retrieval techniques to seed a dense retrieval exploration that uses a document proximity graph. Through extensive experiments, we find that LADR establishes a new dense retrieval effectiveness-efficiency Pareto frontier among approximate k nearest neighbor techniques. When tuned to take around 8ms per query in retrieval latency on our hardware, LADR consistently achieves both precision and recall that are on par with an exhaustive search on standard benchmarks. Importantly, LADR accomplishes this using only a single CPU -- no hardware accelerators such as GPUs -- which reduces the deployment cost of dense retrieval systems.},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {152–162},
numpages = {11},
keywords = {adaptive re-ranking, approximate k nearest neighbor, dense retrieval},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```

## LADR Code

See [here](https://github.com/terrierteam/pyterrier_dr/blob/master/pyterrier_dr/flex/ladr.py) for the code used for running the experiments in the paper.

## Dependencies and Imports

```
import pyterrier as pt
pt.init()
import os
from pyterrier_pisa import PisaIndex
from pyterrier_dr import FlexIndex, TasB
from pyterrier.measures import *
```

## Initializing Retrieval Models

```
bm25 = PisaIndex.from_dataset('msmarco_passage', threads=1).bm25()
model = TasB.dot(batch_size=1, device='cpu') # or other model
```

## Indexing and Building the Corpus Graph

```
idx = FlexIndex('msmarco-passage.tasb.flex')
pipeline = model.doc_encoder() >> idx
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
graph = idx.corpus_graph(128)
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
```

## Defining the Evaluation Function

```
def test(label, p):
  fname = 'ladr_results/' + label.replace('\t', '_') + '.res'
  if not os.path.exists(fname):
    p = p()
    res = p(dataset.get_topics())
    pt.io.write_results(res, fname)
  else:
    res = pt.io.read_results(fname)
  res = pt.Experiment(
      [pt.Transformer.from_df(res)],
      dataset.get_topics(),
      dataset.get_qrels(),
      [nDCG@1000, nDCG@10, R(rel=2)@1000]
  ).iloc[0]
  print(label, res['nDCG@1000'], res['nDCG@10'], res['R(rel=2)@1000'])
  ```

  ## Evaluating BM25, TAS-B, Proactive LADR and Adaptive LADR

  k: no. of neighbors\
  r: seed set size\
  depth: termination criteria

  ```
  test(f'bm25', lambda: bm25)
  test('np', lambda: model.query_encoder() >> idx.np_retriever())
  
  
  for k in [4, 8, 16, 32, 64]:
    for hops in [1]:
      for r in ([10, 20, 50, 100,200,500,1000] if hops == 1 else [10, 20, 50, 100, 200]):
        bm25.num_results = r
        test(f'ladr\tk={k}\thops={hops}\t{r}', lambda: bm25 >> model.query_encoder() >> idx.ladr(k, hops))
  
  for k in [4, 8, 16, 32, 64]:
    for r in [1000]:
      for depth in [1, 2, 5, 10, 20, 50, 100, 200]:
        test(f'adaladr\tk={k}\tr={r}\t{depth}', lambda: bm25 >> model.query_encoder() >> idx.ada_ladr(k, depth=depth))  
  ```

