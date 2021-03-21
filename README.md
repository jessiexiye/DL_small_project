# DL_small_project
Deep Learning course small project: X-ray images based pneumonia for COVID and non-COVID classification

To run the training code:
1. multiclass predictor
```
python multiclass_predictor.py --use_gpu --train
```

2. two binary classifier:
```
python main_predictor.py --use_gpu --train
```

To test the best performing models:
1. multiclass predictor
```
python multiclass_predictor.py --use_gpu --test
```
2. two binary classifier:
```
python main_predictor.py --use_gpu --test
```
