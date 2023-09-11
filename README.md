# Originial repository is tweaked to detect rooftops

Original paper 2019:
```bibtex
@inproceedings{zhang2019ppgnet,
  title={PPGNet: Learning Point-Pair Graph for Line Segment Detection},
  author={Ziheng Zhang and Zhengxin Li and Ning Bi and Jia Zheng and Jinlei Wang and Kun Huang and Weixin Luo and Yanyu Xu and Shenghua Gao},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
Adaptation for rooftops 2021:
```bibtex
 @Article{isprs-archives-XLVI-4-W4-2021-85-2021,
	AUTHOR = {Hensel, S. and Goebbels, S. and Kada, M.},
	TITLE = {BUILDING ROOF VECTORIZATION WITH PPGNET},
	JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
	VOLUME = {XLVI-4/W4-2021},
	YEAR = {2021},
	PAGES = {85--90},
	URL = {https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLVI-4-W4-2021/85/2021/},
	DOI = {10.5194/isprs-archives-XLVI-4-W4-2021-85-2021}
 }
```
Original repositories: https://github.com/SimonHensel/Vectorization-Roof-Data-Set.git & https://github.com/svip-lab/PPGNet.git

(Future) differences with with original repositories:
- Add trained models on roof data set
- Add requirements.txt -> a lot of the old requirements seemed outdated, which ran into trouble with newer GPU
- Add RoofGAN preprocessing, so vectorzation will be easier and more accurate

