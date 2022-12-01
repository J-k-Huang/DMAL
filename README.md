# Balancing Transferability and Discriminability for Unsupervised Domain Adaptation. (TNNLS 2022)
## Environment
* python   3.5.4  
* pytorch  1.0.0  
## Framework
![image](https://github.com/J-k-Huang/DMAL/blob/main/framework.png)
## Benchmark
* Digits contains three datasets: MNIST, USPS, and SVHN. 
* Office-31 contains 4652 images across 31 classes from three domains: Amazon (A), DSLR (D), and Webcam (W). Office-31 dataset can be found [here](https://faculty.cc.gatech.edu/~judy/domainadapt/)
* Office-Home contains 15 500 images of 65 classes from four domains: Ar, Cl, Pr, and Rw. Office-Home dataset can be found [here](https://www.hemanthdv.org/officeHomeDataset.html)
* ImageCLEF-DA contins 12 common classes shared by three domains: C, I, and P.
* VisDA-2017 is a simulation-to-real dataset for domain adaptation over 280 000 images across 12 categories. VisDA-2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public)
* DomainNet consists of about 0.6 million images with 345 classes. DomainNet dataset can be found [here](http://ai.bu.edu/M3SDA/)  
## Demo  
Train on DomainNet 
```  
cd Image_DMAL
python main.py
```
## Citation  
If you use this code for your research, please consider citing:
```
@ARTICLE{9893737,
  author={Huang, Jingke and Xiao, Ni and Zhang, Lei},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Balancing Transferability and Discriminability for Unsupervised Domain Adaptation}, 
  year={2022},
  pages={1-8},
  doi={10.1109/TNNLS.2022.3201623}}  
```
## Contact  
If you have any problem about our code, feel free to contact jkhuang@cqu.edu.cn
