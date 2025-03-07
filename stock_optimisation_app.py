#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model_path = 'stock_optimization_model_gradient.pkl'
model = joblib.load(model_path)

#  DataFrame loaded here for demonstration; in production, this might come from a database.
df_products = pd.read_csv('EnhancedProductData.csv')

@app.route('/predict/<int:product_id>', methods=['GET'])
def predict(product_id):
    try:
        # Fetch the product details by ID
        product = df_products[df_products['ProductId'] == product_id]
        if product.empty:
            return jsonify({'error': 'Product not found'}), 404

        # Assume features are in the correct order
        features = product[['Price', 'CompetitorPrice', 'CostOfGoods', 'SeasonalInfluence', 'MarketingSpend', 'CustomerDemographics','LeadTime', 'ReorderPoint', 'MinimumOrderQuantity', 'HoldingCosts', 'OrderingCosts', 'DemandVariability']].values
        prediction = model.predict(features)
        return jsonify({'product_id': product_id, 'predicted_stock_level': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




