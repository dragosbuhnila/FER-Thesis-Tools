Comparison Task: (note that the order of subtraction and mean-calculation has changed AGAIN. We're back to dif roi by roi, and then mean of these differences) (also you now have aggregation code I guess.)
    - FEDMAR vs MATVIN
    - MARFRO vs MATVIN
    - maschi vs femmine
    - upper tail vs lower tail
    - Bubbles ConvNext vs FEDMAR
    - Bubble ConvNext vs MARFRO
    - External Perturbation ConvNext vs FEDMAR
    - External Perturbation ConvNext vs MARFRO
    - GradCam LAYER30 ConvNext vs FEDMAR
    - GradCam LAYER30 ConvNext vs FEDMAR    


1) ~~Do comparisons with meandif (with lr_separate)~~

2)  ~~Make aggr folder: ~~
    Prepare one folder [for aggregated stat approach] for each emotion (i.e. ANGRY_anger, ANGRY_happiness), in which there'll be the roi-means_vector for GROUP1 and GROUP2 (aggregated), where the groups will be the usual ones of comparisons from the top of this file. Example: (one of these for every versus from above, so 10 in total)
        > ANGRY_ANGRY
            - ANGRY_ANGRY_GROUP1.npy
            - ANGRY_ANGRY_GROUP2.npy
        > ANGRY_HAPPY
            - ANGRY_HAPPY_GROUP1.npy
            - ANGRY_HAPPY_GROUP2.npy
        > ...

3) Prepare one folder [for granular stat approach] for each emotion, in which there'll be ...  Example: (one of these for every versus from above, so 10 in total)
    > ANGRY_ANGRY
        > ANGRY_ANGRY_GROUP1
            - ANGRY_ANGRY_NAME1_mean-vector.npy
            - ANGRY_ANGRY_NAME2_mean-vector.npy
            - ANGRY_ANGRY_NAME3_mean-vector.npy
        > ANGRY_ANGRY_GROUP2
            - ...
    > ANGRY_HAPPY
        > ANGRY_HAPPY_GROUP1
            - ...
        > ANGRY_HAPPY_GROUP2
            - ...
    > ...

