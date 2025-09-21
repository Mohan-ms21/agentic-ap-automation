# AP Policies (Demo)
- Auto-approve invoices < $5,000 from trusted vendors.
- Require 2-way match (PO Amount ~ Invoice Amount within ±2%).
- Require 3-way match if Goods Receipt available; otherwise fallback to 2-way.
- Escalate marketing invoices > $50,000.
- Duplicate rule: (vendor_id + invoice_number) must be unique.
