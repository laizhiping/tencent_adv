# tencent_adv
This repository is for tencent advertising algorithm competition 

## 1. 数据说明

### 1.1 ad.csv

| creative_id | ad_id | product_id | product_category | advertiser_id | industry |
| :----: | :----: | :----:  | :----: | :----: | :----: |
| 素材id[1, 4445720]  | 属于某个广告id[1, 3812202] | 该广告中宣传产品id[1, 44314]| 广告产品所属类别id[1,18] | 广告主id[2, 62965] | 所属行业id[1, 335] |
### 1.2 click_log.csv
| time | user_id | creative_id | click_times |
| :----: | :----: | :----:  | :----: |
| 某一天[1, 91]| 用户id[1, 900000], [3000001,4000000] | 用户点击的素材id[1, 4445720] | 当天该用户点击该素材次数[1, 185] |
### 1.3 user.csv

| user_id | age | gender |
| :----: | :----:  | :----: |
| 用户id | 用户年龄[1, 10] | 用户性别[1, 2] |

## 2. 数据预处理

###  2.1, 连接三张表

| creative_id | ad_id          | product_id         | product_category   | advertiser_id | industry   | time          | user_id | click_times              | age             | gender |
| :-----------: | :--------------: | :------------------: | :------------------: | :-------------: | :----------: | :-------------: | :-------: | :------------------------: | :---------------: | :------: |
| 素材id     | 属于某个广告id | 该广告中宣传产品id | 广告产品所属类别id | 广告主id      | 所属行业id | 某一天 | 用户id  | 当天该用户点击该素材次数 | 用户年龄[1, 10] | 用户性别[1, 2] |



### 2.2 网络输入

| creative_id | ad_id | product_category | advertiser_id |  time | click_times |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 素材id | 属于某个广告id |广告产品所属类别id | 广告主id | 某一天[1, 91] | 当天该用户点击该素材次数 |

### 2.3 网络输出

| age | gender |
| :----:  | :----: |
| 用户年龄[1, 10] | 用户性别[1, 2] |

## 3. 网络模型