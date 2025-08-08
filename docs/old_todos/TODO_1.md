8) Fix the grid code (filename and path issue) {cm:2025-08-04}

9) Test the statistics functions by checking if a completely blue image and a completely white one actually have a difference of one {cm:2025-08-04}

5) Fix the scale of difference to go from 0 to 1 and check if a totally red and a totally blue actually have 1 as difference {cm:2025-08-04}

6) Add the baseface images behind the heatmaps {cm:2025-08-04}
 

1) manually extract (screenshot + reshape) all faces needed for landmarking the au ROIs {cm:2025-08-04}

2) Rebalance the weighing of the ROI stats to take into account areas that are shared a lot by different AUs {cm:2025-08-04}

3) Write a menu to decide what comparison to work on {cm:2025-08-04}
    - Person (name select) {cm:2025-08-04}
        # will compare all canonicals for that person
        # eventually:
        > compare non canonicals too
    - Emotion canonical {cm:2025-08-04}
        # will compare the selected emotion for all subjects
        # eventually:
        > restrict to top % {cm:2025-08-04}
        > restrict to gender + top %
        > restrict some other way
    - Person and Model
        # will compare the selected person with the selected model's layers and XAI methods
        # eventually
        > merge some layers (in some yet to be defined way)

4) Try abandoning the AU approach and going for eyes/mouth/... instead




# Final TODOs
- Modify the save/load functions to push to a specific folder so that I can easily use gitignore and not involve them