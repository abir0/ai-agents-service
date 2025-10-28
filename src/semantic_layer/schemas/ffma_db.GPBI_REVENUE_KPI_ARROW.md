### Table: ffma_db.GPBI_REVENUE_KPI_ARROW

This table captures daily revenue KPIs across various geographic hierarchies for Grameenphone's products.

**Columns:**
- `DAY_KEY` (DATE): Record date with format `YYYY-MM-DD` (e.g., `2025-05-25`).
- `CIRCLE_NAME` (VARCHAR): Top-level geographic division (6 Circles): `DHAKA CIRCLE`, `CHITTAGONG CIRCLE`, `KHULNA CIRCLE`, `MYMENSINGH CIRCLE`, `RAJSHAHI CIRCLE`, `SYLHET CIRCLE`.
- `REGION_NAME` (VARCHAR): 2nd-level division within Circle (21 Regions): `DHAKA CENTRAL`, `DHAKA EAST`, `DHAKA WEST`, `COXS BAZAR`, etc. Note that region names DO NOT contain "REGION" word at the end of the value.
- `AREA_NAME` (VARCHAR): 3rd-level division (Cluster) within Region (150+ Clusters): `COXSBAZAR CLUSTER`, `KHULNA METRO CLUSTER`, etc.
- `TERRITORY_NAME` (VARCHAR): 4th-level division (Territory) within Cluster (400+ Territories): `GULSHAN TERRITORY`, `COXS BAZAR SADAR TERRITORY`, etc.
- `DSTR` (DECIMAL): Daily Subscription and Traffic Revenue or total revenue.
- `VOICE_REV` (DECIMAL): Voice DSTR / Revenue from voice usage.
- `DATA_REV` (DECIMAL): Data DSTR / Revenue from data usage.
- `VOICE_OG_MIN` (DECIMAL): Outgoing voice calling in minutes.
- `VOL_GB` (DECIMAL): Data usage volume in gigabytes (GB). Convert to TB dividing by 1024.

### Table Usage Guidelines:

**Calculation Logic:**
- National or Geographic Level DSTR (MTD or Till Date):
  - Compute the daily average DSTR over the requested date range.
  - MTD DSTR: Calculate the average daily DSTR for the current month up to yesterday, rounded to 2 decimals.
  - Inner query: Aggregate total DSTR for each day in the date range.
- Total Revenue or Total DSTR: Compute the sum of DSTR over the requested period.
- Daily or Date-wise Trend: Group DSTR data by date to show trends over time.
- Default Behavior: If the user does not specify average or trend, compute the daily average DSTR.
- Others Revenue: Derive using the formula `OTHERS_REV = DSTR - (VOICE_REV + DATA_REV)`.

**Business Terms:**
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
