## DaRe
http://proceedings.mlr.press/v139/brophy21a.html
13 publicly-available datasets, 1 synthetic dataset we call Synthetic.
For each dataset, we generate **one-hot encodings** for any categorical variable and leave all numeric and binary variables as is. For any dataset without a designated train and test split, we **randomly** sample **80%** of the data for training and use the rest for testing.

A summary of the datasets is in Table 1, and additional dataset details are in the Appendix: §B.1.

![[Pasted image 20240722102223.png]]


#### Surgical
Kaggle. Dataset surgical binary classification. https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification/version/1#, 2018c. [Online; accessed 29-July-2020].

- Surgical (Kaggle, 2018c) consists of 14,635 medical patient surgeries (3,690 positive cases), characterized by 25 attributes; the goal is to predict whether or not a patient had a complication from their surgery.

#### Vaccine
Bull, P., Slavitt, I., and Lipstein, G. Harnessing the power of the crowd to increase capacity for data science in the social sector. In *ICML # Data4Good Workshop*, 2016.

DrivenData. Flu shot learning: Predict h1n1 and seasonal flu vaccines. https://www.drivendata.org/competitions/66/flu-shot-learning/data/, 2019. [Online; accessed 12-August-2020].

- Vaccine (Bull et al., 2016; DrivenData, 2019) consists of 26,707 survey responses collected between October 2009 and June 2010 asking people a range of 36 behavioral and personal questions, and ultimately asking whether or not they got an H1N1 and/or seasonal flu vaccine. Our aim is to predict whether or not a person received a seasonal flu vaccine.

#### Adult
Becker,Barry and Kohavi,Ronny. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.

Dua, D. and Graff, C. UCI machine learning repository. http://archive.ics.uci.edu/ml, 2019.

- Adult (Dua & Graff, 2019) contains 48,842 instances (11,687 positive) of 14 demographic attributes to determine if a person’s personal income level is more than $50K per year.

https://archive.ics.uci.edu/dataset/2/adult

#### Bank Marketing
Moro, S., Cortez, P., et al. A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 2014.

Dua, D. and Graff, C. UCI machine learning repository. http://archive.ics.uci.edu/ml, 2019.
\
- Bank Marketing (Moro et al., 2014; Dua & Graff, 2019) consists of 41,188 marketing phone calls (4,640 positive) from a Portuguese banking institution. There are 20 attributes, and the aim is to figure out if a client will subscribe.

https://archive.ics.uci.edu/dataset/222/bank+marketing


#### Flight Delays
Research and Administration, I. T. Airline on-time performance and causes of flight delays. https://catalog.data.gov/dataset/airline-on-time-performance-and-causes-of-flight-delays-on-time-data, 2019. [Online; accessed 16-April-2020].

- Flight Delays (Research & Administration, 2019) consists of 100,000 actual arrival and departure times of flights by certified U.S. air carriers; the data was collected by the Bureau of Transportation Statistics’ (BTS) Office of Airline Information. The data contains 8 attributes and 19,044 delays. The task is to predict if a flight will be delayed.

#### Diabetes
Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

Dua, D. and Graff, C. UCI machine learning repository. http://archive.ics.uci.edu/ml, 2019.

- Diabetes (Strack et al., 2014; Dua & Graff, 2019) consists of 101,766 instances of patient and hospital readmission outcomes (46,902 readmitted) characterized by 55 attributes.

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
**What do the instances in this dataset represent?**
The instances represent hospitalized patient records diagnosed with diabetes.

**Are there recommended data splits?**
No recommendation. The standard train-test split could be used. Can use three-way holdout split (i.e., train-validation-test) when doing model selection.

**Does the dataset contain data that might be considered sensitive in any way?**
Yes. The dataset contains information about the age, gender, and race of the patients.

**Additional Information**
The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria. (1) It is an inpatient encounter (a hospital admission). (2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered into the system as a diagnosis. (3) The length of stay was at least 1 day and at most 14 days. (4) Laboratory tests were performed during the encounter. (5) Medications were administered during the encounter. The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab tests performed, HbA1c test result, diagnosis, number of medications, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

**Has Missing Values?**
Yes

#### No Show
Kaggle. Medical appointment no shows. https://www.kaggle.com/joniarroba/noshowappointments, 2016. [Online; accessed 25-Januaray-2021].

- NoShow(Kaggle, 2016) contains 110,527 instances of patient attendances for doctors’ appointments (22,319 no shows) characterized by 14 attributes. The aim is to predict whether or not a patient shows up to their doctors’ appointment.

#### Olympics
Kaggle. 120 years of olympic history: Athletes and events. https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results, 2018b. [Online; accessed 28-July- 2020].

- Olympics (Kaggle, 2018b) contains 206,165 Olympic events over 120 years of Olympic history. Each event contains information about the athlete, their country, which Olympics the event took place, the sport, and what type of medal the athlete received. The aim is to predict whether or not an athlete received a medal for each event they participated in.

#### Census
Dua, D. and Graff, C. UCI machine learning repository. http://archive.ics.uci.edu/ml, 2019.

- Census (Dua & Graff, 2019) contains 40 demographic and employment attributes on 299,285 people in the United States; the survey was conducted by the U.S. Census Bureau. The goal is to predict if a person’s income level is more than $50K.

#### Credit Card
Kaggle. Credit card fraud detection. https://www.kaggle.com/mlg-ulb/creditcardfraud/, 2018a. [Online; accessed 27-July-2020].

- Credit Card (Kaggle, 2018a) contains 284,807 credit card transactions in September 2013 by European cardholders. The transactions took place over two days and contains 492 fraudulent charges (0.172% of all charges). There are 28 principal components resulting from PCA on the original dataset, and two additional fetures: ‘time’ and ‘amount’. The aim is to predict whether a charge is fraudulent or not.

#### Click-Through Rate (CTR)
Criteo. Criteo click-through rate prediction. https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/, 2015. [Online; accessed 25-Januaray-2021].

- Click-Through Rate (CTR) (Criteo, 2015) contains the first 1,000,000 instances of the Criteo 1TB Click Logs dataset, in which each row represents an ad that was displayed and whether or not it had been clicked on (29,040 ads clicked). The dataset contains 13 numeric attributes and 26 categorical attributes. However, due to the extremely large number of values for the categorical attributes, we restrict our use of the dataset to the 13 numeric attributes. The aim is to predict whether or not an ad is clicked on.

*Failed to download*

#### Twitter
Sedhai, S. and Sun, A. Hspam14: A collection of 14 million tweets for hashtag-oriented spam research. In SIGIR, 2015.

- Twitter uses the first 1,000,000 tweets (169,471 spam) of the HSpam14 dataset (Sedhai & Sun, 2015). Each instance contains the tweet ID and label. After retrieving the text and user ID for each tweet, we derive the following attributes: no. chars, no. hashtags, no. mentions, no. links, no. retweets, no. unicode chars., and no. messages per user. The aim is to predict whether a tweet is spam or not.

https://personal.ntu.edu.sg/axsun/datasets.html

*Contact the author to get a smaller version*
#### Synthetic
Pedregosa, F., Varoquaux, G., Gramfort, A., et al. Scikit-learn: Machine learning in Python. JMLR, 2011.

- Synthetic (Pedregosa et al., 2011) contains 1,000,000 instances normally distributed about the vertices of a 5-dimensional hypercube into 2 clusters per class. There are 5 informative attributes, 5 redundant attributes, and 30 useless attributes. There is interdependence between these attributes, and a randomly selected 5% of the labels are flipped to increase the difficulty of the classification task.

#### Higgs
Baldi, P., Sadowski, P., and Whiteson, D. Searching for exotic particles in high-energy physics with deep learning. Nature Communications, 2014.

Dua, D. and Graff, C. UCI machine learning repository. http://archive.ics.uci.edu/ml, 2019.

- Higgs (Baldi et al., 2014; Dua & Graff, 2019) contains 11,000,000 signal processes (5,829,123 Higgs bosons) characterized by 22 kinematic properties measured by detectors in a particle accelerator and 7 attributes derived from those properties. The goal is to distinguish between a background signal process and a Higgs bosons process.

## HedgeCut
#### Income ("Adult income")
The same as the "Adult" in DaRe.
This dataset contains 390K data points in 32,560 records of demographic and financial data, with four numerical and eight categorical attributes, and the target variable denotes whether a person earns more than 50,000 dollars per year or not.
 https://archive.ics.uci.edu/ml/datasets/Adult
#### Heart Disease
This dataset contains 770K data points in 70,000 patient records comprised of five numerical and six categorical measurements with respect to cardiovascular diseases, and the target variable denotes the presence of a heart disease.
https://www.kaggle.com/sulianova/cardiovascular-disease-dataset
#### Credit ("Credit Card")

https://www.kaggle.com/c/GiveMeSomeCredit
#### Recidivism 

#### Purchase Data ("Online purchase behavior data")
