
## Deciphering Environmental Air Pollution (DEAP)

Official repository for the paper "Deciphering Environmental Air Pollution with Large Scale City Data", IJCAI 2022



[comment]:<> (Data for the paper "Deciphering Environmental Air Pollution with Large Scale City Data")

### Main Dataset
**city_pollution_data.csv**

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

    
### Power Plant Generation and Location Dataset [Extra]:
**pp_gen_data.csv**

Relevant Columns:

* `Month`: Month of the data
* `Netgen`: Net generation for that month.

Project Website: https://mayukh18.github.io/DEAP/
