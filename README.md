
  
## Deployment
Overall I'm aiming to simply provide a more straightforward (and better :)) process to rooftop vectorisation. 
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

### Enhancing edges
This is done automatically when running ```bash test.sh [testfilename]```.
However you can also run ```python edge_enhancer.py``` if you simply want to 'draw' the edges on top of the image:
Without arguments: It will process all images in the specified directories.
With arguments: It will process only the image pair given as arguments.
For processing all images: ```python edge_enhancer.py --train_or_val=train```
For processing a specific image pair: ```python edge_enhancer.py --edge=path_to_edge_img --original=path_to_original_img --output=path_to_save_enhanced_img```

## Credits
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

Differences with with original repositories:
- Add trained models on roof data set
- Add requirements.txt -> a lot of the old requirements seemed outdated, which ran into trouble with newer GPU
- Add extra datasets: edges are 'drawn' on top, which are ready to use for transfer learning, so vectorzation will be easier and more accurate
- Add trained models on enhanced roof data set
- Resize images, lines and coordinates to original size
- Keep logs of coordinates for developers

Edge detection:
```bibtex
@InProceedings{xie15hed,
  author = {"Xie, Saining and Tu, Zhuowen"},
  Title = {Holistically-Nested Edge Detection},
  Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
  Year  = {2015},
}
```

Edge thinning:
https://stackoverflow.com/a/35815510
