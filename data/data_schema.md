# FIFA 2026 Group-Stage Data Schema

Six CSVs, all CSVs join on `team_id` or `venue_id`.

---

## 1. `matches.csv`

| Column | Type | Description |
|---|---|---|
| `match_id` | int | Match number in ascending kickoff order (1–72) |
| `group` | str | Group letter (A–L) |
| `round` | int | Match round within group (1, 2, or 3) |
| `team_a_id` | str | Team ID of first team (joins → `teams.csv`) |
| `team_b_id` | str | Team ID of second team (joins → `teams.csv`) |
| `venue_id` | str | Venue identifier (joins → `venues.csv`) |
| `date` | date | Match date (YYYY-MM-DD) |
| `kickoff_local` | time | Kickoff time in venue local time (HH:MM) |

---

## 2. `venues.csv`

| Column | Type | Description |
|---|---|---|
| `venue_id` | str | Unique venue identifier (e.g. `DAL`, `NYC`, `MEX`) |
| `name` | str | Stadium name |
| `city` | str | Host city |
| `country` | str | Host country (`USA`, `CAN`, `MEX`) |
| `lat` | float | Latitude (decimal degrees) |
| `lon` | float | Longitude (decimal degrees) |
| `utc_offset_june` | int | UTC offset during June (accounts for daylight saving) |
| `zone` | str | Geographic cluster for group-stage scheduling (`Western`, `Central`, or `Eastern`) |

---

## 3. `teams.csv`

| Column | Type | Description |
|---|---|---|
| `team_id` | str | Unique team identifier (e.g. `BRA`, `GER`, `MEX`) |
| `team_name` | str | Full team name |
| `group` | str | Assigned group (A–L) |
| `fifa_ranking` | int | FIFA Men's World Ranking (lower = stronger) |

---

## 4. `base_camps.csv`

62 FIFA candidate facilities (IDs 1–62) plus 17 confirmed team base camps that fell outside the candidate list (IDs 63–79). Confirmed assignments carry a `team_id`; unassigned candidates do not.

| Column | Type | Description |
|---|---|---|
| `base_camp_id` | int | Unique facility ID; 1–62 match FIFA candidate list numbering, 63–79 are confirmed assignments outside that list |
| `team_id` | str\|null | FIFA 3-letter team code if assigned, else empty (joins → `teams.csv`) |
| `training_site` | str | Specific facility name (e.g. `Gonzaga University`, `Waters Sportsplex`) |
| `city` | str | Municipality or region |
| `country` | str | Host country (`USA` or `Mexico`) |
| `lat` | float | Facility-level latitude, decimal degrees (Nominatim-geocoded where available, else hard-coded approximation) |
| `lon` | float | Facility-level longitude, decimal degrees |
| `utc_offset_june` | int | UTC offset during June (DST-adjusted; Arizona stays at −7 year-round; Mexico at −6 since abolishing DST in 2022) |


---

## 5. `weather.csv` — 6,528 rows

Hourly temperature for every venue across the full group-stage window. Query by `venue_id` + `datetime` at match kickoff time (or average over match window).

| Column | Type | Description |
|---|---|---|
| `venue_id` | str | Venue identifier (joins → `venues.csv`) |
| `datetime` | datetime | Hourly UTC timestamp in 2026 (YYYY-MM-DD HH:MM) |
| `temperature_c` | float | 3-year average air temperature at 2 m height (°C), rounded to 1 decimal |

---

## 6. `broadcast_markets.csv` — 48 rows

One row per qualified nation. Each row links a country to its competing team, local prime-time window, UTC offset, and population-based audience weight.

| Column | Type | Description |
|---|---|---|
| `country` | str | Country name |
| `team_id` | str | Corresponding team ID if a qualified nation; `null` otherwise |
| `primetime_start_local` | time | Start of prime-time window in country's local time (HH:MM) |
| `primetime_end_local` | time | End of prime-time window in country's local time (HH:MM) |
| `utc_offset_june` | int | Country's UTC offset during June (accounts for daylight saving) |
| `audience_weight` | int | Population (or viewership estimate) used to weight broadcast value |

---

## Join Map

```
matches ──── venue_id ──────────────► venues
matches ──── team_a_id / team_b_id ──► teams
teams   ──── team_id ───────────────► base_camps
venues  ──── venue_id + datetime ───► weather
venues  ──── utc_offset_june ───────► broadcast_markets (prime-time conversion)
broadcast_markets ── team_id ───────► teams (to link market to a qualified nation)
```

---

## ID Legend

### Venue IDs

| `venue_id` | Stadium | City | Country | Zone |
|---|---|---|---|---|
| `ATL` | Mercedes-Benz Stadium | Atlanta | USA | Eastern |
| `BOS` | Gillette Stadium | Boston | USA | Eastern |
| `DAL` | AT&T Stadium | Dallas | USA | Central |
| `GDL` | Estadio Akron | Guadalajara | MEX | Central |
| `HOU` | NRG Stadium | Houston | USA | Central |
| `KC` | GEHA Field at Arrowhead Stadium | Kansas City | USA | Central |
| `LAX` | SoFi Stadium | Los Angeles | USA | Western |
| `MEX` | Estadio Azteca | Mexico City | MEX | Central |
| `MIA` | Hard Rock Stadium | Miami | USA | Eastern |
| `MTY` | Estadio BBVA | Monterrey | MEX | Central |
| `NYC` | MetLife Stadium | New York/New Jersey | USA | Eastern |
| `PHI` | Lincoln Financial Field | Philadelphia | USA | Eastern |
| `SFO` | Levi's Stadium | San Francisco Bay Area | USA | Western |
| `SEA` | Lumen Field | Seattle | USA | Western |
| `TOR` | BMO Field | Toronto | CAN | Eastern |
| `VAN` | BC Place | Vancouver | CAN | Western |

---

### Team IDs

| `team_id` | Team | Group |
|---|---|---|
| `ALG` | Algeria | J |
| `ARG` | Argentina | J |
| `AUS` | Australia | D |
| `AUT` | Austria | J |
| `BEL` | Belgium | G |
| `BIH` | Bosnia & Herzegovina | B |
| `BRA` | Brazil | C |
| `CAN` | Canada | B |
| `CIV` | Côte d'Ivoire | E |
| `COD` | DR Congo | K |
| `COL` | Colombia | K |
| `CPV` | Cape Verde | H |
| `CRO` | Croatia | L |
| `CUW` | Curaçao | E |
| `CZE` | Czechia | A |
| `ECU` | Ecuador | E |
| `EGY` | Egypt | G |
| `ENG` | England | L |
| `ESP` | Spain | H |
| `FRA` | France | I |
| `GER` | Germany | E |
| `GHA` | Ghana | L |
| `HAI` | Haiti | C |
| `IRN` | IR Iran | G |
| `IRQ` | Iraq | I |
| `JOR` | Jordan | J |
| `JPN` | Japan | F |
| `KOR` | South Korea | A |
| `KSA` | Saudi Arabia | H |
| `MAR` | Morocco | C |
| `MEX` | Mexico | A |
| `NED` | Netherlands | F |
| `NOR` | Norway | I |
| `NZL` | New Zealand | G |
| `PAN` | Panama | L |
| `PAR` | Paraguay | D |
| `POR` | Portugal | K |
| `QAT` | Qatar | B |
| `RSA` | South Africa | A |
| `SCO` | Scotland | C |
| `SEN` | Senegal | I |
| `SUI` | Switzerland | B |
| `SWE` | Sweden | F |
| `TUN` | Tunisia | F |
| `TUR` | Türkiye | D |
| `URU` | Uruguay | H |
| `USA` | United States | D |
| `UZB` | Uzbekistan | K |
