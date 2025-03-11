
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

file_path = "rupa_107_decision_tree.csv"
df = pd.read_csv(file_path)

# Drop missing values in 'Items' column
df_cleaned = df.dropna(subset=['Items'])

# Group by 'AppointmentID' to form transactions
transactions = df_cleaned.groupby('AppointmentID')['Items'].apply(list).tolist()

# Convert transactions to one-hot encoding
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# Check item frequencies
print("Item Frequencies:\n", df_encoded.sum().sort_values(ascending=False))

# Apply Apriori with very low min_support
freq_items = apriori(df_encoded, min_support=0.005, use_colnames=True)

# Print frequent itemsets
print("Frequent Itemsets:\n", freq_items)

# Apply Association Rules with lower confidence
if not freq_items.empty:
    final_association = association_rules(freq_items, metric="confidence", min_threshold=0.3)

    # If confidence gives no rules, try lift
    if final_association.empty:
        final_association = association_rules(freq_items, metric="lift", min_threshold=1.0)

    print("Association Rules:\n", final_association)
    final_association.to_csv("final_output.csv", index=False)
else:
    print("❌ No frequent itemsets found. Try lowering min_support further.")


final_association = association_rules(freq_items, metric="lift", min_threshold=1.0)
final_association.sort_values(by=["confidence", "lift"], ascending=[False, False], inplace=True)
final_association = association_rules(freq_items, metric="confidence", min_threshold=0.2)

print(final_association[['antecedents', 'consequents', 'confidence', 'lift']].to_string())
