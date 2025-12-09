#%%import mesa
import random
import numpy as np
import pandas as pd

# --- GLOBAL CONSTANTS (From Report Section 3.1 & 3.5) ---

# Financial Constants
COST_PER_LABEL_UPGRADE = 5000  # Cost to jump one energy label
MOVING_COST_FIXED = 2000       # Fixed cost of relocating
MOVING_COST_VARIABLE = 0.02    # e.g., 2% of house value

# Weights for Satisfaction Equation (The alpha, beta, gamma from Eq 3.2)
WEIGHT_FINANCIAL = 0.4
WEIGHT_COMFORT = 0.3
WEIGHT_ENVIRONMENT = 0.3

# Normalization Bounds (Needed for Eq 3.2 to keep values between 0-1)
MIN_WEALTH = 0
MAX_WEALTH = 500000
MAX_QUALITY = 1.0
MAX_SIZE = 200  # square meters

#%% --- HELPER FUNCTIONS ---

def normalize(value, min_val, max_val):
    """
    Normalizes a value to 0-1 scale.
    """
    if max_val == min_val:
        return 0
    # Clamp value to ensure it stays between 0-1
    val = max(min_val, min(value, max_val))
    return (val - min_val) / (max_val - min_val)

def calculate_mortgage_capacity(income):
    """
    Simple rule to determine max mortgage (e.g., 5x income).
    """
    return income * 5

def compute_avg_label(model):
    """
    DataCollector function to track average energy label.
    Checks all HomeOwner agents and averages their house's energy label.
    """
    agent_labels = [a.house.energy_label for a in model.schedule.agents 
                    if isinstance(a, HomeOwner)]
    if not agent_labels: return 0
    return sum(agent_labels) / len(agent_labels)

    

#%% Assets (house agent)
class House(mesa.Agent):
    def __init__(self, unique_id, model, energy_label, size, quality, price):
        super().__init__(unique_id, model)
        self.energy_label = energy_label  # Int 0-7 (G to A+++)
        self.size = size
        self.quality = quality
        self.market_value = price
        
        # Linkage
        self.owner = None    # Will point to a HomeOwner object
        self.is_vacant = False 

    def update_energy_label(self, levels=1):
        """
        Increments label and updates market value.
        """
        self.energy_label += levels
        # Assuming value increases with energy label
        self.market_value += (levels * 2000)

#%% The decision maker (HomeOwner agent)

class HomeOwner(mesa.Agent):
    def __init__(self, unique_id, model, income, savings, house):
        super().__init__(unique_id, model)
        self.income = income
        self.savings = savings
        self.house = house
        
        # Decision Variables
        self.intention = "STAY" 
        self.target_house_id = None

    # ADDED: bonus_energy parameter
    def calculate_satisfaction(self, target_house, upgrade_cost=0, moving_cost=0, bonus_energy=0):
        """
        Calculates Utility (S). 
        bonus_energy: Int used to simulate an upgrade (e.g. +1 label)
        """
        # 1. Projected Wealth
        projected_savings = self.savings - upgrade_cost - moving_cost
        norm_financial = normalize(projected_savings, MIN_WEALTH, MAX_WEALTH)
        
        # 2. House Attributes (WITH BONUS ADDED)
        # We clamp the value to max 7 using min(..., 7)
        projected_label = min(target_house.energy_label + bonus_energy, 7)
        
        norm_energy = normalize(projected_label, 0, 7)
        norm_quality = normalize(target_house.quality, 0, MAX_QUALITY)
        
        # 3. Weighted Sum
        S = (WEIGHT_FINANCIAL * norm_financial) + \
            (WEIGHT_ENVIRONMENT * norm_energy) + \
            (WEIGHT_COMFORT * norm_quality)
        
        return S

    def step(self):
        # --- A. EVALUATE CURRENT SITUATION (STAY) ---
        S_current = self.calculate_satisfaction(self.house)
        
        # --- B. EVALUATE UPGRADE (STAY + RENOVATE) ---
        S_upgrade = -999 # Default low value
        
        # Only check if they can afford it AND the house isn't already maxed out (Label < 7)
        if self.savings > COST_PER_LABEL_UPGRADE and self.house.energy_label < 7:
            # FIX: We pass bonus_energy=1 to simulate the benefit
            S_upgrade = self.calculate_satisfaction(self.house, 
                                                  upgrade_cost=COST_PER_LABEL_UPGRADE,
                                                  bonus_energy=1)
        
        # --- C. EVALUATE MARKET (MOVE) ---
        best_market_house = None
        S_move = -999
        
        for house in self.model.vacancy_pool:
            if house.market_value <= (self.savings + calculate_mortgage_capacity(self.income)):
                 s_temp = self.calculate_satisfaction(house, moving_cost=MOVING_COST_FIXED)
                 if s_temp > S_move:
                     S_move = s_temp
                     best_market_house = house
        
        # --- D. COMPARE AND SET INTENTION ---
        if S_move > S_upgrade and S_move > S_current:
            self.intention = "BUY"
            self.target_house_id = best_market_house.unique_id
        elif S_upgrade > S_current:
            self.intention = "UPGRADE"
            self.target_house_id = None
        else:
            self.intention = "STAY"
            self.target_house_id = None
            
        # Optional: Print intention for debugging
        # print(f"{self.unique_id}: {self.intention} (Stay={S_current:.2f}, Upg={S_upgrade:.2f}, Move={S_move:.2f})")

#%% Market manager

class HousingModel(mesa.Model):
    def __init__(self, n_agents=10, n_houses=15):
        super().__init__()
        # FIX: Removed "advance" from stage_list to avoid errors
        self.schedule = mesa.time.StagedActivation(self, stage_list=["step"]) 
        self.vacancy_pool = []
        self.datacollector = mesa.DataCollector(
            model_reporters={"Avg_Label": compute_avg_label}
        )
        
        # FIX: Create a dictionary to look up houses by ID later
        self.house_map = {} 

        # 1. Create HOUSES
        all_houses = []
        for i in range(n_houses):
            h = House(unique_id=f"H_{i}", 
                      model=self, 
                      energy_label=random.randint(0, 5), 
                      size=random.randint(50, 150), 
                      quality=random.random(), 
                      price=random.randint(100000, 300000))
            all_houses.append(h)
            self.house_map[h.unique_id] = h # Store in map
        
        # 2. Create OWNERS
        for i in range(n_agents):
            house = all_houses[i]
            owner = HomeOwner(unique_id=f"Owner_{i}", 
                              model=self, 
                              income=random.randint(30000, 80000), 
                              savings=random.randint(5000, 50000), 
                              house=house)
            house.owner = owner
            self.schedule.add(owner)
        
        # 3. Handle Remaining Houses
        for i in range(n_agents, n_houses):
            vacant_house = all_houses[i]
            vacant_house.is_vacant = True
            self.vacancy_pool.append(vacant_house)
            
        print(f"Model Initialized: {n_agents} Agents, {len(self.vacancy_pool)} Vacant Houses")

    def resolve_market_conflicts(self):
        # 1. Collect all bids FIRST
        bids = {}
        for agent in self.schedule.agents:
            if isinstance(agent, HomeOwner) and agent.intention == "BUY":
                target = agent.target_house_id
                if target not in bids:
                    bids[target] = []
                bids[target].append(agent)
        
        # Debug Print
        if len(bids) > 0:
            print(f"MARKET: {len(bids)} houses have bids.")

        # 2. Resolve Bids
        for house_id, potential_buyers in bids.items():
            # Sort by Financial Capacity (Savings + Max Mortgage)
            winner = sorted(potential_buyers, 
                          key=lambda x: x.savings + calculate_mortgage_capacity(x.income), 
                          reverse=True)[0]
            
            print(f"MARKET: House {house_id} sold to {winner.unique_id}")
            self.execute_move(winner, house_id)

    def execute_move(self, buyer, house_id):
        # 1. Buyer leaves old house
        old_house = buyer.house
        old_house.owner = None
        old_house.is_vacant = True
        self.vacancy_pool.append(old_house)
        
        # 2. Buyer gets new house (Using the FIX: self.house_map)
        new_house = self.house_map[house_id] 
        new_house.owner = buyer
        new_house.is_vacant = False
        
        if new_house in self.vacancy_pool:
            self.vacancy_pool.remove(new_house)
            
        # 3. Update Buyer
        buyer.savings -= MOVING_COST_FIXED
        buyer.house = new_house

    def step(self):
        self.schedule.step()
        self.resolve_market_conflicts()
        
        # Handle Upgrades
        for agent in self.schedule.agents:
            if isinstance(agent, HomeOwner) and agent.intention == "UPGRADE":
                agent.house.update_energy_label()
                agent.savings -= COST_PER_LABEL_UPGRADE
                agent.intention = "STAY"
        
        self.datacollector.collect(self)

#%% --- RUN TEST SIMULATION ---
# 1. Setup
print("--- STARTING SIMULATION ---")
model = HousingModel(n_agents=5, n_houses=8)

# 2. Run for 3 steps
for i in range(3):
    print(f"\n--- STEP {i+1} ---")
    model.step()

# 3. Check Results
print("\n--- FINAL RESULTS ---")
print(f"Average Energy Label: {compute_avg_label(model):.2f}")
for agent in model.schedule.agents:
    if isinstance(agent, HomeOwner):
        print(f"Agent {agent.unique_id} is in House {agent.house.unique_id} (Label: {agent.house.energy_label})")