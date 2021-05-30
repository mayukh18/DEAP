# Modeling-Air-Pollution
Data for the paper "Modeling the Dynamics of Air Pollution: A Bayesian Approach on Large Scale Data"

1.  Main Dataset: aqi_city_data.csv

    Relevant Columns:

    * Date: Date of the sample
    * City: City of the sample
    * Specie: Type of pollutant
    * Median: Median value of the pollutant for the day 
    * mil_miles: Total vehicle travel distance for the sample
    * past_week_avg_miles: Average vehicle travel distance for last 7 days
    * pp_feat: Calculated feature for the influence of neighboring power plants
    
2.  Power Plant Generation and Location Dataset: pp_gen_data.csv

    Relevant Columns:

    * Month: Month of the data
    * Netgen: Net generation for that month.