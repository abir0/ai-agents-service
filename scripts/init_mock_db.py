import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))  # noqa: E402

import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel

from src.database import AsyncPostgresManager


# Pydantic models for data validation
class ProductSpecs(BaseModel):
    cpu: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None
    display: Optional[str] = None
    gpu: Optional[str] = None
    battery: Optional[str] = None
    connectivity: Optional[str] = None
    sensors: Optional[List[str]] = None
    water_resistance: Optional[str] = None
    features: Optional[List[str]] = None


class Product(BaseModel):
    id: str
    name: str
    category: str
    price: float
    stock: int
    description: str
    specs: ProductSpecs
    created_at: str
    is_available: bool


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip: str
    country: str


class CustomerPreferences(BaseModel):
    newsletter: bool
    notifications: List[str]
    favorite_categories: List[str]


class Customer(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    created_at: str
    address: Address
    preferences: CustomerPreferences


class OrderItem(BaseModel):
    product_id: str
    quantity: int
    unit_price: float
    total: float


class Payment(BaseModel):
    method: str
    status: str
    total_amount: float


class Order(BaseModel):
    id: str
    customer_id: str
    created_at: str
    status: str
    items: List[OrderItem]
    shipping_address: Address
    payment: Payment
    notes: str


# Sample data for different types of documents
SAMPLE_PRODUCTS = [
    {
        "id": "prod_001",
        "name": "Laptop Pro X1",
        "category": "electronics",
        "price": 1299.99,
        "stock": 50,
        "description": "High-performance laptop for professionals",
        "specs": {
            "cpu": "Intel i7 12th Gen",
            "ram": "32GB DDR5",
            "storage": "1TB NVMe SSD",
            "display": '15.6" 4K OLED',
            "gpu": "NVIDIA RTX 4060",
        },
        "created_at": datetime.now().isoformat(),
        "is_available": True,
    },
    {
        "id": "prod_002",
        "name": "SmartWatch Pro",
        "category": "wearables",
        "price": 299.99,
        "stock": 75,
        "description": "Advanced fitness and health tracking",
        "specs": {
            "display": '1.4" AMOLED',
            "battery": "48 hours",
            "sensors": ["heart rate", "ECG", "SpO2"],
            "water_resistance": "5ATM",
        },
        "created_at": datetime.now().isoformat(),
        "is_available": True,
    },
    {
        "id": "prod_003",
        "name": "Wireless Earbuds Plus",
        "category": "audio",
        "price": 159.99,
        "stock": 100,
        "description": "Premium wireless audio experience",
        "specs": {
            "battery": "30 hours total",
            "connectivity": "Bluetooth 5.2",
            "features": ["ANC", "Transparency Mode", "Touch Controls"],
        },
        "created_at": datetime.now().isoformat(),
        "is_available": True,
    },
]

SAMPLE_CUSTOMERS = [
    {
        "id": "cust_001",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1-555-0123",
        "created_at": datetime.now().isoformat(),
        "address": {
            "street": "123 Tech Lane",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94105",
            "country": "USA",
        },
        "preferences": {
            "newsletter": True,
            "notifications": ["email", "sms"],
            "favorite_categories": ["electronics", "wearables"],
        },
    },
    {
        "id": "cust_002",
        "name": "Emma Wilson",
        "email": "emma.wilson@example.com",
        "phone": "+1-555-0124",
        "created_at": datetime.now().isoformat(),
        "address": {
            "street": "456 Innovation Drive",
            "city": "Austin",
            "state": "TX",
            "zip": "78701",
            "country": "USA",
        },
        "preferences": {
            "newsletter": False,
            "notifications": ["email"],
            "favorite_categories": ["audio"],
        },
    },
]

SAMPLE_ORDERS = []
# Generate sample orders
for i in range(10):
    order_date = datetime.now() - timedelta(days=random.randint(1, 30))
    customer = random.choice(SAMPLE_CUSTOMERS)
    product = random.choice(SAMPLE_PRODUCTS)
    quantity = random.randint(1, 3)

    SAMPLE_ORDERS.append(
        {
            "id": f"ord_{i + 1:03d}",
            "customer_id": customer["id"],
            "created_at": order_date.isoformat(),
            "status": random.choice(["pending", "processing", "shipped", "delivered"]),
            "items": [
                {
                    "product_id": product["id"],
                    "quantity": quantity,
                    "unit_price": product["price"],
                    "total": round(quantity * product["price"], 2),
                }
            ],
            "shipping_address": customer["address"],
            "payment": {
                "method": random.choice(["credit_card", "paypal", "bank_transfer"]),
                "status": "completed",
                "total_amount": round(
                    quantity * product["price"] + 10.00, 2
                ),  # Including shipping
            },
            "notes": "Handle with care",
        }
    )


async def init_mock_db():
    """Initialize the mock database with sample data."""
    db_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/aiagents"

    async with AsyncPostgresManager(db_url) as manager:
        # Create tables using Pydantic models
        await manager.create_table_from_pydantic(Product, "products")
        await manager.create_table_from_pydantic(Customer, "customers")
        await manager.create_table_from_pydantic(Order, "orders")

        # Insert sample data
        print("Inserting sample products...")
        for product in SAMPLE_PRODUCTS:
            await manager.create_data("products", product)

        print("Inserting sample customers...")
        for customer in SAMPLE_CUSTOMERS:
            await manager.create_data("customers", customer)

        print("Inserting sample orders...")
        for order in SAMPLE_ORDERS:
            await manager.create_data("orders", order)

        # Verify the data
        print("\nVerifying inserted data:")
        for table_name in ["products", "customers", "orders"]:
            docs = await manager.read_all_data(table_name)
            print(f"{table_name}: {len(docs)} documents")


if __name__ == "__main__":
    asyncio.run(init_mock_db())
