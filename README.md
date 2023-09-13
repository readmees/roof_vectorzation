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
- Add extra preprocessing and use those preprocessed images for more transfer learning, so vectorzation will be easier and more accurate
- Resize images, lines and coordinates to original size
- Keep logs of coordinates for developers
  
## Deployment
Overall I'm aiming to simply provide a more straightforward process to rooftop vectorisation. 
Pipeline to setup (requires Miniconda):
```bash
git clone https://github.com/readmees/roof_vectorzation.git
cd roof_vectorzation
sudo apt-get install git-lfs
bash install.sh
bash test.sh [testfilename]
```
test.sh simply calls test.py, so you can integrate test.py into your existing Python scripts.
If you want to retrain the model on the dataset run: 
```bash
bash train.sh
```
In case you would like to use your own dataset, simply change the "dataset path in the train.sh script (modify the --data-root parameter)" step 3 of https://github.com/svip-lab/PPGNet.git 
