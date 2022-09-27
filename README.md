# Deciphering Environmental Air Pollution (DEAP)

Official repository for the paper [Deciphering Environmental Air Pollution with Large Scale City Data](https://www.ijcai.org/proceedings/2022/0698.pdf), IJCAI 2022

1. [Paper Overview](https://mayukh18.github.io/DEAP/). 
2. [Alternate PyTorch implementation](https://github.com/sayannag/cossquareformer-pytorch) based on [davidsvy's](https://github.com/davidsvy/cosformer-pytorch) cosFormer implementation..

[comment]:<> (Data for the paper "Deciphering Environmental Air Pollution with Large Scale City Data")

## Main Dataset
`city_pollution_data.csv`

Relevant Columns:

* `Date`: Date of the sample
* `City`: City of the sample
* `X_median`: Median value of the pollutant/meteorological feature X for the day 
* `mil_miles`: Total vehicle travel distance for the sample
* `pp_feat`: Calculated feature for the influence of neighboring power plants
* `Population Staying at Home`: Used a measure of domestic emissions.

**Pollutants**:
`PM2.5`,`PM10`,`NO2`,`O3`,`CO`,`SO2`

**Meteorological Features**:
`Temperature`,`Pressure`,`Humidity`,`Dew`,`Wind Speed`,`Wind Gust`

    
## Power Plant Generation and Location Dataset [Extra]:
`pp_gen_data.csv`

Relevant Columns:

* `Month`: Month of the data
* `Netgen`: Net generation for that month.

If you find the data or code useful in your work, please cite
```
@inproceedings{ijcai2022p698,
  title     = {Deciphering Environmental Air Pollution with Large Scale City Data},
  author    = {Bhattacharyya, Mayukh and Nag, Sayan and Ghosh, Udita},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  year      = {2022},
}
```
