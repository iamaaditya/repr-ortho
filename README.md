# repr-ortho
Repr training and inter-filter ranking

### Data
https://drive.google.com/open?id=1q-Hfm7vKMi-7M0NUaS3VmJ1yqzlZrP-d
Checksum: 3914026746

To train with repr:
`python train.py`
To train with standard:
`python train.py --standard`.


## Notes

1. Baseline training accuracy overfits, this is because RePr was designed to improve upon overfitting of small and vanilla conv-nets. 
2. Baseline test accuracy might appear low. It is because of two reasons: 
    (i) There is no augmentation. RePr was designed to show overfitting of even small network
    (ii) Only uses 90% of the training data. Other is used for validation.

