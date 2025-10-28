### Table: ffma_db.GPBI_ERS_STS_ROW_HISTORY_ALL

This table captures hourly telecom recharge and service usage data, categorized by geographic location, sales executives, and the service/packs attributes.

**Columns:**
- `DAY_KEY` (DATE): Record date with format `YYYY-MM-DD` (e.g., `2025-05-25`).
- `CIRCLE_NAME` (VARCHAR): Top-level geographic division (6 Circles): `DHAKA CIRCLE`, `CHITTAGONG CIRCLE`, `KHULNA CIRCLE`, `MYMENSINGH CIRCLE`, `RAJSHAHI CIRCLE`, `SYLHET CIRCLE`.
- `REGION_NAME` (VARCHAR): 2nd-level division within Circle (21 Regions): `DHAKA CENTRAL`, `DHAKA EAST`, `DHAKA WEST`, `COXS BAZAR`, etc. Note that region names DO NOT contain "REGION" word at the end of the value.
- `AREA_NAME` (VARCHAR): 3rd-level division (Cluster) within Region (150+ Clusters): `COXSBAZAR CLUSTER`, `KHULNA METRO CLUSTER`, etc.
- `TERRITORY_NAME` (VARCHAR): 4th-level division (Territory) within Cluster (400+ Territories): `GULSHAN TERRITORY`, `COXS BAZAR SADAR TERRITORY`, etc.
- `SE_CD` (VARCHAR): Sales Executive code (5th-level within Territory, 4000+ SEs) (e.g., `SESRC020288`).
- `SERVICE_TYPE` (VARCHAR): Service type: `ERS_STS` (ERS_STS = TR + PL + RC + RELOAD), `TR` (Trigger), `PL` (Power Load), `RC` (Rate Cutter), or `RELOAD` (Basic/vanilla recharge).
- `SERVICE_NAME` (VARCHAR): Sub-category under SERVICE_TYPE: `ERS_STS`, `COMBO`, `DATA`, `VOICE`, `OTHER`, `SMS`, `RC`, or `REGULAR`.
- `RECHARGE_DENOM` (NUMBER): Recharge denomination/deno or pack category in BDT (e.g., `29`, `99`, `898`); only applicable for service types: PL, TR, or RC.
- `HITS` (NUMBER): Total count of transactions for the service or denomination.
- `AMT` (NUMBER): Total sales in BDT (`AMT = RECHARGE_DENOM * HITS`).
- `GP_SKITTO` (VARCHAR): Platform or user type of GP: `GP` (Regular user), or `SKITTO` (Special app user).
- `SE_NAME` (VARCHAR): Sales Executive full name (e.g., `Md. Rahim Uddin`).

### Table Usage Guidelines:

**SERVICE_TYPE -> SERVICE_NAME Mapping:**

Each `SERVICE_TYPE` has specific allowed `SERVICE_NAME` values:
# SERVICE_TYPE -> SERVICE_NAME #
- `ERS_STS` -> `ERS_STS`.
- `TR` -> `COMBO`/`DATA`/`VOICE`/`OTHER`.
- `PL` -> `COMBO`/`DATA`/`SMS`/`VOICE`.
- `RC` -> `RC`.
- `RELOAD` -> `REGULAR`.

**SERVICE_TYPE, SERVICE_NAME, or RECHARGE_DENOM Filtering Rules:**
- Use `SERVICE_TYPE`, `SERVICE_NAME`, and `RECHARGE_DENOM` filters only if the user mentions:
  - a service category (TR, RC, etc.)
  - a pack type (DATA, VOICE, etc.)
  - a recharge denomination (e.g. deno 29, 99 tk).
- If the user asks about a recharge denomination, price, or pack:
  - Restrict `SERVICE_TYPE` to only: `TR` or `PL`. So use: `SERVICE_TYPE IN ('TR','PL')`.
  - If the amount is not specified: use `RECHARGE_DENOM IS NOT NULL`.
  - If the amount is specified (e.g., "deno 29", "99 tk pack"): use RECHARGE_DENOM = [value].
  - "Bundle pack" or "Bundle pack STS" corresponds to `COMBO`.
  - "Pack" and "Pack STS" are equivalent.

**Date Handling:**
- If `DAY_KEY` is in SELECT then you MUST use `ORDER BY DAY_KEY DESC`.
- If no reference date is mentioned, get current date using function.
- Date filters to use:
  - Today: `WHERE DAY_KEY = CURDATE()`.
  - Last day or yesterday: `WHERE DAY_KEY = CURDATE() - INTERVAL 1 DAY`.
  - Last month: `WHERE DAY_KEY BETWEEN (CURDATE() - INTERVAL 1 MONTH - INTERVAL DAYOFMONTH(CURDATE())-1 DAY) AND LAST_DAY(CURDATE() - INTERVAL 1 MONTH)`.
  - MTD (Month-till-Date) or Till yesterday: `WHERE DAY_KEY BETWEEN DATE_FORMAT(CURDATE(), '%Y-%m-01') AND CURDATE()`.
  - LMTD (Last Month-till-Date): `WHERE DAY_KEY BETWEEN DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m-01') AND CURDATE() - INTERVAL 1 MONTH`.
  - Date range example:
    ```sql
    SELECT DAY_KEY, HITS
    FROM [table_name]
    WHERE DAY_KEY BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
    ORDER BY DAY_KEY DESC
    ```

**Common Business Metrics:**
- *Performance*: Use `SUM(AMT)` and `SUM(HITS)`
- *Contribution*: Use percentage calculations using analytic functions
- *Growth*: Compare current vs. previous period metrics (e.g. MTD, LMTD)

**Business Terms:**
- STS is Sales to Subscriber.
- ERS_STS is the aggregated service type that includes all recharge types: TR + PL + RC + RELOAD.
- TR (Trigger) packs are ATL or above-the-line packs which means recharging certain amounts activates trigger packs for everyone.
- PL (Power Load) packs are BTL or below-the-line packs, these are usually personalized packs.
- RC (Rate Cutter) packs offers reduced call/data rates.
- Data STS refers to data packs (SERVICE_NAME = 'DATA').
- Voice STS refers to voice packs (SERVICE_NAME = 'VOICE').
- DHFF (Distribution House Field Force) refers to Sales Executives.
- Distribution house (DH) refers to territory.

**Default Alias:**
- Use these alias names exactly for consistency in query results:
  - Use `TOTAL_HITS` alias for `SUM(HITS)`.
  - Use `TOTAL_AMT` alias for `SUM(AMT)`.
  - Use `AVG_HITS` alias for `AVG(HITS)`.
  - Use `AVG_AMT` alias for `AVG(AMT)`.

**Default LIMITs:**
- Use the following LIMITs unless otherwise specified:
  - Circle wise data: `LIMIT 6`.
  - Region wise data: `LIMIT 21`.
  - Area/Cluster wise data: `LIMIT 50`.
  - Territory wise data: `LIMIT 50`.
  - SE wise data: `LIMIT 15`.
  - Show Month-wise or MTD data: `LIMIT 31`.
