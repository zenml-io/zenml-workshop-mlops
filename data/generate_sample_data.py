#!/usr/bin/env python3
"""
Generate synthetic customer churn dataset for ZenML workshop
This creates a realistic dataset with logical relationships between features and churn.
"""

import pandas as pd
import numpy as np
import os


def generate_customer_churn_data(n_samples=2000):
    """Generate synthetic customer churn dataset with realistic patterns."""
    np.random.seed(42)

    # Generate base demographic data
    age = np.random.normal(45, 15, n_samples).clip(18, 80).astype(int)
    tenure_months = np.random.exponential(24, n_samples).clip(1, 72).astype(int)

    # Generate financial data with some correlation to demographics
    base_charges = np.random.uniform(20, 150, n_samples)
    monthly_charges = (
        base_charges + (age - 40) * 0.3 + np.random.normal(0, 10, n_samples)
    )
    monthly_charges = monthly_charges.clip(15, 200)

    total_charges = monthly_charges * tenure_months + np.random.normal(
        0, 100, n_samples
    )
    total_charges = total_charges.clip(100, 15000)

    # Generate service usage
    num_services = np.random.poisson(3, n_samples).clip(1, 8)

    # Categorical variables with realistic distributions
    contract_type = np.random.choice(
        ["month-to-month", "one-year", "two-year"],
        n_samples,
        p=[0.55, 0.25, 0.20],  # Month-to-month is most common
    )

    payment_method = np.random.choice(
        ["electronic_check", "credit_card", "bank_transfer", "mailed_check"],
        n_samples,
        p=[0.35, 0.25, 0.25, 0.15],
    )

    # Internet service type
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.4, 0.45, 0.15]
    )

    # Create the base dataframe
    data = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:06d}" for i in range(n_samples)],
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charges": np.round(monthly_charges, 2),
            "total_charges": np.round(total_charges, 2),
            "num_services": num_services,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
        }
    )

    # Create churn based on realistic business logic
    churn_probability = np.zeros(n_samples)

    # Contract type impact (month-to-month customers churn more)
    churn_probability += np.where(data["contract_type"] == "month-to-month", 0.35, 0.0)
    churn_probability += np.where(data["contract_type"] == "one-year", 0.15, 0.0)
    churn_probability += np.where(data["contract_type"] == "two-year", 0.05, 0.0)

    # Tenure impact (newer customers churn more)
    churn_probability += np.where(data["tenure_months"] < 6, 0.25, 0.0)
    churn_probability += np.where(data["tenure_months"] < 12, 0.15, 0.0)
    churn_probability += np.where(data["tenure_months"] > 48, -0.1, 0.0)

    # Price sensitivity
    churn_probability += np.where(data["monthly_charges"] > 80, 0.15, 0.0)
    churn_probability += np.where(data["monthly_charges"] > 120, 0.1, 0.0)

    # Payment method impact (electronic check customers churn more)
    churn_probability += np.where(
        data["payment_method"] == "electronic_check", 0.15, 0.0
    )
    churn_probability += np.where(data["payment_method"] == "mailed_check", 0.1, 0.0)

    # Age impact (younger customers are more likely to switch)
    churn_probability += np.where(data["age"] < 30, 0.1, 0.0)
    churn_probability += np.where(data["age"] > 65, -0.05, 0.0)

    # Service complexity (too many or too few services increase churn)
    churn_probability += np.where(data["num_services"] == 1, 0.1, 0.0)
    churn_probability += np.where(data["num_services"] > 6, 0.05, 0.0)

    # Add some random noise
    churn_probability += np.random.uniform(-0.1, 0.1, n_samples)

    # Ensure probabilities are in valid range
    churn_probability = churn_probability.clip(0, 1)

    # Generate actual churn labels
    data["churned"] = (np.random.random(n_samples) < churn_probability).astype(int)

    return data


def main():
    """Generate and save the customer churn dataset."""
    print("Generating customer churn dataset...")

    # Generate the data
    data = generate_customer_churn_data(n_samples=2500)

    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save to CSV
    output_path = "data/customer_churn.csv"
    data.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Shape: {data.shape}")
    print(f"Churn rate: {data['churned'].mean():.1%}")
    print("\nSample data:")
    print(data.head())

    print("\nChurn distribution by contract type:")
    print(data.groupby("contract_type")["churned"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
