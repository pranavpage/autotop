# Automated Map Generation

## ROUGH
[Q] should the functions for preparing the data be written from scratch?
[A] yeah do it, nothing else is really intensive
[DONE]
## Actual Documentation starts here


## TODOs
1. create single pipeline for burning masks to images, can pass buffer, dissolve as parameters [P3]
Why? to reduce code size and provide modularity 
2. OOP style processing and prediction, with one class for buildings, one class for roads, etc [P3]
3. train model with appropriate loss function [P1]
    a.  new loss is topology based, look at apls metric for inspiration, measure of connectivity desired [P2]
    b. compare with dice, Jaccard [P2]
4. post processing pipeline [P1] 
5. package into app [P1]

## Stretch TODOs
1. stitch given .tifs into a map and show roads, buildings 

