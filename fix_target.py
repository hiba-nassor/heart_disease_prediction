import pandas as pd

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data['target'] = train_data['target'].apply(lambda x: 1 if x > 0 else 0)
test_data['target'] = test_data['target'].apply(lambda x: 1 if x > 0 else 0)

print("Train target distribution after fix:")
print(train_data['target'].value_counts())
print("\nTest target distribution after fix:")
print(test_data['target'].value_counts())

train_data.to_csv('data/train.csv', index=False)
test_data.to_csv('data/test.csv', index=False)
print("\nFiles updated successfully: data/train.csv and data/test.csv")