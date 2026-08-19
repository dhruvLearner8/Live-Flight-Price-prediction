# Airport Code Reference

The full dataset only contains these 16 origin airports. The 10 marked with
✅ are the ones selected for `flight_sample.csv` (chosen because their
routes to/from each other have the highest combined traffic volume in the
dataset, meaning enough historical rows per route for good features).

| Code | Airport Name                                | City              | In sample? |
|------|----------------------------------------------|-------------------|------------|
| ATL  | Hartsfield-Jackson Atlanta International      | Atlanta, GA       | ✅ |
| LAX  | Los Angeles International                     | Los Angeles, CA   | ✅ |
| LGA  | LaGuardia Airport                              | New York, NY      | ✅ |
| BOS  | Logan International                            | Boston, MA        | ✅ |
| JFK  | John F. Kennedy International                  | New York, NY      | ✅ |
| DFW  | Dallas/Fort Worth International                | Dallas/Fort Worth, TX | ✅ |
| ORD  | O'Hare International                           | Chicago, IL       | ✅ |
| DTW  | Detroit Metropolitan Wayne County              | Detroit, MI       | ✅ |
| EWR  | Newark Liberty International                   | Newark, NJ        | ✅ |
| CLT  | Charlotte Douglas International                | Charlotte, NC     | ✅ |
| SFO  | San Francisco International                    | San Francisco, CA | — |
| MIA  | Miami International                            | Miami, FL         | — |
| PHL  | Philadelphia International                     | Philadelphia, PA  | — |
| DEN  | Denver International                           | Denver, CO        | — |
| OAK  | Oakland International                          | Oakland, CA       | — |
| IAD  | Washington Dulles International                | Washington, D.C. (Dulles, VA) | — |

## The 20 routes in the sample (10 city-pairs, both directions)

| Route pair                | Combined rows in full dataset |
|----------------------------|-------------------------------|
| Atlanta <-> Los Angeles    | 1,379,418 |
| Los Angeles <-> New York (LGA) | 1,341,372 |
| Boston <-> Los Angeles     | 1,323,559 |
| New York (JFK) <-> Los Angeles | 1,230,513 |
| Dallas/Fort Worth <-> Los Angeles | 1,223,059 |
| Los Angeles <-> Chicago    | 1,218,423 |
| Detroit <-> Los Angeles    | 1,183,559 |
| Newark <-> Los Angeles     | 1,131,737 |
| Charlotte <-> Los Angeles  | 1,126,571 |
| New York (LGA) <-> Chicago | 1,096,379 |

Note: 9 of the 10 pairs include LAX as one endpoint (LAX is the
highest-traffic origin airport in this dataset). This was a known
tradeoff, kept intentionally since it reflects real air-traffic
patterns and each route still has 1M+ rows of history.
