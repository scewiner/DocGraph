# DocGraph

Document Graph for Nerual Machine Translation, EMNLP 2021

---

### Citation

Please cite as:

```
@inproceedings{xu-etal-2021-document-graph,
    title = "Document Graph for Neural Machine Translation",
    author = "Xu, Mingzhou  and
      Li, Liangyou  and
      Wong, Derek F.  and
      Liu, Qun  and
      Chao, Lidia S.",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.663",
    doi = "10.18653/v1/2021.emnlp-main.663",
}
```

---

### Usage:

1. We release the training scripts:
   1. Start-B.sh is the training script for our  **Post-Integration**
   2. Start.sh for **Pre-Integration**
   3. Start-H.sh for **Hyb-Integration**
2. We release our Zh-En data in examples dir. 
   1. Note that Since the completed data is too large, we only release toy data. This data is without duplicate words, which means each vertex represents all the duplicate words.
   2. We have evaluated the effectiveness of the toy data.  Using this toy data also can achieve a similar translation performance (you can get the models in examples dir). 
3. For some reason, we only finished the test of the code of src-graph.  We will release  the code of the tgt-graph after we finish the test.