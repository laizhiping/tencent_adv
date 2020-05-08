# tencent_adv
This repository is for tencent advertising algorithm competition 
## 1. 数据说明
### 1.1 click_log.csv
| time | user_id | creative_id | click_times |
| :----: | :----: | :----:  | :----: |
| 某一天[1, 91]| 用户id | 用户点击的素材id | 当天该用户点击该素材次数 |
### 1.2 user.csv
| user_id | age | gender |
| :----: | :----:  | :----: |
| 用户id | 用户年龄[1, 10] | 用户性别[1, 2] |
### 1.3 ad.csv
| creative_id | ad_id | product_id | product_category | advertiser_id | industry id |
| :----: | :----: | :----:  | :----: | :----: | :----: |
| 素材id | 属于某个广告id | 该广告中宣传产品id| 广告产品所属类别id | 广告主id | 所属行业id |

## 2. 数据预处理
### 2.1 网络输入
| creative_id | ad_id | product_id | product_category | advertiser_id | industry id | click_times | time |
| :----: | :----: | :----:  | :----: | :----: | :----: | :----: | :----: |
| 素材id | 属于某个广告id | 该广告中宣传产品id | 广告产品所属类别id | 广告主id | 所属行业id | 当天该用户点击该素材次数 | 某一天[1, 91]|

### 2.2 网络输出
| age | gender |
| :----:  | :----: |
| 用户年龄[1, 10] | 用户性别[1, 2] |

## 3. 网络模型
