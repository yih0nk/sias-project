"""
env/customer_model.py
=====================
Customer demand and discrete-choice model (Section 2.2).

Each potential trip request has a pickup zone and a dropoff node.
For each (company, vehicle_type) alternative, the customer computes a
logit utility and samples a choice via softmax.

Equations from the paper
------------------------
Utility of option (c, vt):
    U(c, vt) = -β_price · price - β_wait · wait - β_tt · travel_time   (Eq. 8)

Price of option (c, vt):
    price = (base_fare + per_min_rate · travel_time) · multiplier[c][vt][zone]  (Eq. 9)

There are 4 alternatives: {A_hv, A_av, B_hv, B_av}.
No outside option — every customer picks one.

Note: β_price enters as a COST, so higher price → lower utility.
"""

import numpy as np
from typing import Dict, List, Tuple

from config import (
    N_COMPANIES, VEHICLE_TYPES, COMPANY_NAMES,
    BETA_PRICE, BETA_WAIT, BETA_TT,
    BASE_FARE, PER_MIN_RATE,
)


class Request:
    """
    One potential trip from a customer.

    Parameters
    ----------
    request_id : int
    pickup_zone : int       TLC zone index (0..74)
    dropoff_zone : int
    pickup_edge : str       SUMO edge at pickup location
    dropoff_edge : str      SUMO edge at dropoff location
    arrival_epoch : int     When the request appeared
    travel_time_est : float Estimated travel time (seconds) on base network
    """

    _counter = 0

    def __init__(
        self,
        pickup_zone: int,
        dropoff_zone: int,
        pickup_edge: str,
        dropoff_edge: str,
        arrival_epoch: int,
        travel_time_est: float,
    ):
        Request._counter += 1
        self.request_id       = Request._counter
        self.pickup_zone      = pickup_zone
        self.dropoff_zone     = dropoff_zone
        self.pickup_edge      = pickup_edge
        self.dropoff_edge     = dropoff_edge
        self.arrival_epoch    = arrival_epoch
        self.travel_time_est  = travel_time_est   # seconds

        # Filled in after choice
        self.chosen_company   = None   # int 0/1
        self.chosen_vtype     = None   # "hv" or "av"
        self.assigned_vid     = None   # vehicle id string

    @classmethod
    def reset_counter(cls):
        cls._counter = 0


class CustomerModel:
    """
    Handles demand generation and customer choice.

    Parameters
    ----------
    demand_data : list[dict]
        Pre-loaded list of potential trips per epoch. Each entry is a dict:
            { 'epoch': int, 'pickup_zone': int, 'dropoff_zone': int,
              'pickup_edge': str, 'dropoff_edge': str }
        In practice this comes from TLC FHV records (see data_loader.py).
    rng : np.random.Generator
    """

    def __init__(self, demand_data: List[dict], rng: np.random.Generator):
        self.demand_data = demand_data
        self.rng         = rng

        # Build an index: epoch → list of raw trip dicts
        self._epoch_index: Dict[int, List[dict]] = {}
        for trip in demand_data:
            e = trip["epoch"]
            self._epoch_index.setdefault(e, []).append(trip)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_requests_for_epoch(
        self,
        epoch: int,
        travel_times: Dict[str, float],
    ) -> List[Request]:
        """
        Convert raw trip records for this epoch into Request objects with
        estimated travel times looked up from the current SUMO edge speeds.

        Parameters
        ----------
        epoch : int
        travel_times : dict
            Maps edge_id → current travel time (seconds). Obtained from
            TrafficInterface.get_edge_travel_times() after each epoch.

        Returns
        -------
        list[Request]
        """
        raw_trips = self._epoch_index.get(epoch, [])
        requests = []
        for trip in raw_trips:
            # Estimate trip travel time: use known edge travel time if available,
            # otherwise fall back to free-flow estimate stored in the record.
            tt_est = trip.get("travel_time_ff", 600.0)   # default 10 min
            # If we have a live edge measurement for the pickup edge, use it
            if trip["pickup_edge"] in travel_times:
                tt_est = travel_times[trip["pickup_edge"]]

            req = Request(
                pickup_zone      = trip["pickup_zone"],
                dropoff_zone     = trip["dropoff_zone"],
                pickup_edge      = trip["pickup_edge"],
                dropoff_edge     = trip["dropoff_edge"],
                arrival_epoch    = epoch,
                travel_time_est  = tt_est,
            )
            requests.append(req)
        return requests

    def customer_choice(
        self,
        request: Request,
        price_multipliers: np.ndarray,
        wait_times: np.ndarray,
    ) -> Tuple[int, str]:
        """
        Apply logit choice model (Eq. 8-9) to select a (company, vtype) pair.

        Parameters
        ----------
        request : Request
        price_multipliers : np.ndarray, shape (N_COMPANIES, 2)
            [company, vtype_idx] → price multiplier.
            vtype_idx: 0=hv, 1=av
        wait_times : np.ndarray, shape (N_COMPANIES, 2)
            Estimated wait time in seconds for each (company, vtype).

        Returns
        -------
        (company_idx, vtype_str)
        """
        tt_min = request.travel_time_est / 60.0   # convert to minutes

        utilities = []
        for c in range(N_COMPANIES):
            for vt_idx, vtype in enumerate(VEHICLE_TYPES):
                mult  = price_multipliers[c, vt_idx]
                price = (BASE_FARE + PER_MIN_RATE * tt_min) * mult
                wait  = wait_times[c, vt_idx]

                # Eq. 8 (negative because higher values = worse for customer)
                u = (- BETA_PRICE * price
                     - BETA_WAIT  * wait
                     - BETA_TT    * tt_min)
                utilities.append(u)

        # Softmax over 4 alternatives
        utilities  = np.array(utilities, dtype=np.float64)
        exp_u      = np.exp(utilities - utilities.max())   # numerical stability
        probs      = exp_u / exp_u.sum()

        choice_idx = self.rng.choice(len(probs), p=probs)

        # Decode flat index → (company, vtype)
        company    = choice_idx // len(VEHICLE_TYPES)
        vt_idx     = choice_idx %  len(VEHICLE_TYPES)
        return company, VEHICLE_TYPES[vt_idx]

    # ── Zone-level summaries (used for observations) ──────────────────────────

    def compute_zone_demand(
        self,
        requests: List[Request],
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        total_demand : int
        zone_inflow  : np.ndarray shape (N_ZONES,)  — how many trips END here
        zone_outflow : np.ndarray shape (N_ZONES,)  — how many trips START here
        """
        from config import N_ZONES
        total    = len(requests)
        inflow   = np.zeros(N_ZONES, dtype=np.float32)
        outflow  = np.zeros(N_ZONES, dtype=np.float32)
        for r in requests:
            outflow[r.pickup_zone]  += 1
            inflow[r.dropoff_zone]  += 1
        if total > 0:
            inflow  /= total
            outflow /= total
        return total, inflow, outflow
