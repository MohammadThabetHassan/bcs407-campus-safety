# Why Campus Safety Monitoring Matters

## The Problem Is Real and Urgent

Campus safety is not an abstract concern — it is a **daily operational challenge** faced by universities, colleges, and school districts worldwide. Every year, preventable incidents on campuses result in injuries, property damage, regulatory violations, and even fatalities.

### Key Statistics

| Category | Data Point | Source |
|----------|-----------|--------|
| **Campus fires** | ~3,800 structure fires per year in U.S. dormitories alone | NFPA, "Structure Fires in Dormitories" |
| **Slip/fall injuries** | Leading cause of workplace injury; wet floors are a primary contributor | OSHA Injury Statistics |
| **Fire alarm failures** | 22% of fire deaths occur in properties with non-functioning alarms | NFPA Report (2023) |
| **PPE non-compliance** | ~65% of workplace eye injuries occur because workers were not wearing proper PPE | BLS Survey |
| **Cost of non-compliance** | Average OSHA penalty: $15,625 per serious violation; $156,259 per willful/repeat | OSHA Penalty Schedule |
| **Campus crime** | ~28,000 criminal incidents reported on U.S. campuses annually | Clery Act Data (2022) |

### Why Manual Monitoring Falls Short

1. **Human fatigue**: Security staff monitoring CCTV feeds lose attention after 20–30 minutes (research shows detection accuracy drops 45% after 30 minutes of continuous monitoring).
2. **Infrequent inspections**: Most campuses conduct fire safety and PPE compliance checks weekly or monthly — violations can persist for days.
3. **Coverage gaps**: A single campus may have 200+ fire alarm stations and 500+ safety signs; physically inspecting all of them is impractical.
4. **Delayed response**: By the time a human notices a missing safety helmet in a construction zone or a blocked emergency exit, an accident may have already occurred.

### The Opportunity

Computer vision systems can provide **continuous, real-time, automated monitoring** of safety conditions:

- Detect whether fire alarms are present and unobstructed
- Verify wet floor signs are placed after cleaning
- Confirm emergency exit signs are visible and correctly positioned
- Monitor PPE (safety helmets) compliance in construction/workshop zones

This system bridges the gap between periodic manual inspections and the need for continuous safety assurance.

### Regulatory Context

| Standard | Relevance |
|----------|-----------|
| **OSHA 29 CFR 1910** | Workplace safety requirements including PPE, fire extinguisher placement, hazard communication |
| **NFPA 101 Life Safety Code** | Requirements for emergency exit signage, fire alarm systems |
| **NFPA 72** | National Fire Alarm and Signaling Code |
| **ISO 45001** | Occupational health and safety management systems |
| **Canadian OH&S** | Provincial occupational health and safety regulations |
| **Clery Act** (US) | Campus crime reporting and safety disclosure requirements |

Non-compliance with any of these standards can result in fines, accreditation loss, liability lawsuits, and — most importantly — preventable harm to students and staff.

### Project Goal

This project demonstrates that **YOLOv8-based object detection** can be applied to automatically identify and monitor four critical safety objects in campus environments:

1. **Wet floor signs** — slip-and-fall prevention
2. **Fire alarms** — fire emergency readiness
3. **Emergency exit signs** — evacuation route compliance
4. **Safety helmets** — PPE compliance monitoring

By automating detection of these objects, the system enables:
- Real-time compliance dashboards
- Instant alerts for missing or obstructed safety equipment
- Historical trend analysis of safety compliance
- Reduced reliance on manual inspections