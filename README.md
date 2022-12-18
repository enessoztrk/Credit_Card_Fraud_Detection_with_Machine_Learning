# Credit Card Fraud Detection with Machine Learning
- The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features.

## What is Fraud Detection?
![img](https://github.com/enessoztrk/Fraud_Detection_with_Machine_Learning/blob/main/img/fraud.PNG?raw=true)

- Fraud detection protects person information, assets, accounts and transactions through the real-time, near-real-time analysis of activities by users and other defined entities. It uses background server-based processes that examine users’ and other defined entities’ access and behavior patterns, and typically compares this information to a profile of what’s expected.
- Project is live now visit at https://www.kaggle.com/code/enesztrk/fraud-detection-with-machine-learning
​
## Data Description:
- TransactionDT: Timedelta from a given reference datetime (not an actual timestamp)
- TransactionAMT: Transaction payment amount in USD
- ProductCD: Product code, the product for each transaction
- card1 - card6: Payment card information, such as card type, card category, issue bank, country, etc.
- addr: Address
- dist: Distance
- P_ and R_emaildomain: Purchaser and recipient email domain
- C1-C14: Counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
- D1-D15: Timedelta, such as days between previous transaction, etc.
- M1-M9: mMatch, such as names on card and address, etc.
- Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

- Categorical Features: ProductCD, card1 - card6, addr1, addr2, P_emaildomain, R_emaildomain, M1 - M9
    
## Identity Table:
- Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. They're collected by Vesta’s fraud protection system and digital security partners. (The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)

- Categorical Features: DeviceType, DeviceInfo, id_12 - id_38

![img1](https://github.com/enessoztrk/Fraud_Detection_with_Machine_Learning/blob/main/img/fraud1.PNG?raw=true)
​
