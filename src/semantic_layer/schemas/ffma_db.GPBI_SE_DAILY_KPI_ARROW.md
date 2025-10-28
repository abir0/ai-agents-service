### Table: ffma_db.GPBI_SE_DAILY_KPI_ARROW

This table tracks daily performance KPIs of Sales Executives (SEs) across various geographic hierarchies for Grameenphone's products.

**Columns:**
- `DAY_KEY` (DATE): Record date with format `YYYY-MM-DD` (e.g., `2025-05-25`).
- `CIRCLE_NAME` (VARCHAR): Top-level geographic division (6 Circles): `DHAKA CIRCLE`, `CHITTAGONG CIRCLE`, `KHULNA CIRCLE`, `MYMENSINGH CIRCLE`, `RAJSHAHI CIRCLE`, `SYLHET CIRCLE`.
- `REGION_NAME` (VARCHAR): 2nd-level division within Circle (21 Regions): `DHAKA CENTRAL`, `DHAKA EAST`, `DHAKA WEST`, `COXS BAZAR`, etc. Note that region names DO NOT contain "REGION" word at the end of the value.
- `AREA_NAME` (VARCHAR): 3rd-level division (Cluster) within Region (150+ Clusters): `COXSBAZAR CLUSTER`, `KHULNA METRO CLUSTER`, etc.
- `TERRITORY_NAME` (VARCHAR): 4th-level division (Territory) within Cluster (400+ Territories): `GULSHAN TERRITORY`, `COXS BAZAR SADAR TERRITORY`, etc.
- `SE_CD` (VARCHAR): Sales Executive code (5th-level within Territory, 4000+ SEs) (e.g., `SESRC020288`).
- `GA` (DECIMAL): Gross Add/Addition (total number of customers acquired through SIM card sales).
- `SKITTO_GA` (DECIMAL): Gross Add for Skitto platform users.
- `TOTAL_POS` (DECIMAL): Total number of active Point-of-Sale outlets under the SE's coverage.
- `ERS_STR` (DECIMAL): KPI for Sales to Retailer.
- `UPDATED_AT` (DATE): Timestamp or date when this KPI record was last updated.
- `SE_NAME` (VARCHAR): Sales Executive full name (e.g., `Md. Rahim Uddin`).

### Table Usage Guidelines:

**Business Terms:**
- STR means Sales to Retailer.
- Distribution house (DH) refers to territory.

**Date Handling:**
- The table contains data up to the previous day (i.e. Day-1 or D-1). Use `CURDATE() - INTERVAL 1 DAY` when querying the most recent day.
- If `DAY_KEY` is in SELECT then you MUST use `ORDER BY DAY_KEY DESC`.
- If no reference date is mentioned, get current date using function.
- Date filters to use:
  - Last day or yesterday or no date mentioned: `WHERE DAY_KEY = CURDATE() - INTERVAL 1 DAY`.
  - Last month (full month data): `WHERE DAY_KEY BETWEEN (CURDATE() - INTERVAL 1 MONTH - INTERVAL DAYOFMONTH(CURDATE())-1 DAY) AND LAST_DAY(CURDATE() - INTERVAL 1 MONTH)`.
  - MTD (Month-till-Date) or Till yesterday: `WHERE DAY_KEY BETWEEN DATE_FORMAT(CURDATE(), '%Y-%m-01') AND CURDATE()`.
  - LMTD (Last Month-till-Date): `WHERE DAY_KEY BETWEEN DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m-01') AND CURDATE() - INTERVAL 1 MONTH`.
  - Date range example:
    ```sql
    SELECT DAY_KEY, HITS
    FROM [table_name]
    WHERE DAY_KEY BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
    ORDER BY DAY_KEY DESC
    ```

**Default LIMITs:**
- Use the following LIMITs unless otherwise specified:
  - Circle wise data: `LIMIT 6`.
  - Region wise data: `LIMIT 21`.
  - Area/Cluster wise data: `LIMIT 50`.
  - Territory wise data: `LIMIT 50`.
  - Show Month-wise or MTD data: `LIMIT 31`.
