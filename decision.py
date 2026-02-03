"""
=============================================================================
DECISION MAKING MODULE - Enhanced Fuzzy Logic Control System
=============================================================================
22 comprehensive fuzzy rules for intelligent energy management with
advanced membership functions and decision explanations.

Author: MATOUSSI Tasnim
Lab: AI-P1 | ISET Bizerte
=============================================================================
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EnhancedFuzzyEnergyController:
    """
    Advanced Fuzzy Logic Controller with 22 comprehensive rules organized
    into 6 categories for optimal energy management.
    """
    
    def __init__(self):
        self._create_fuzzy_variables()
        self._create_fuzzy_rules()
        self.ctrl_system = ctrl.ControlSystem(self.rules)
        self.controller = ctrl.ControlSystemSimulation(self.ctrl_system)
        self.decision_log = []
        
    def _create_fuzzy_variables(self):
        """Create fuzzy input and output variables with enhanced membership functions"""
        
        # ==================== INPUT VARIABLES ====================
        
        # Energy Demand (0-5000W) - 5 levels
        self.energy_demand = ctrl.Antecedent(np.arange(0, 5001, 1), 'energy_demand')
        self.energy_demand['very_low'] = fuzz.trimf(self.energy_demand.universe, [0, 0, 500])
        self.energy_demand['low'] = fuzz.trimf(self.energy_demand.universe, [300, 1000, 1700])
        self.energy_demand['medium'] = fuzz.trimf(self.energy_demand.universe, [1500, 2500, 3500])
        self.energy_demand['high'] = fuzz.trimf(self.energy_demand.universe, [3000, 4000, 5000])
        self.energy_demand['very_high'] = fuzz.trimf(self.energy_demand.universe, [4500, 5000, 5000])
        
        # Solar Generation (0-3000W) - 5 levels
        self.solar_gen = ctrl.Antecedent(np.arange(0, 3001, 1), 'solar_gen')
        self.solar_gen['none'] = fuzz.trimf(self.solar_gen.universe, [0, 0, 100])
        self.solar_gen['poor'] = fuzz.trimf(self.solar_gen.universe, [50, 400, 800])
        self.solar_gen['moderate'] = fuzz.trimf(self.solar_gen.universe, [700, 1500, 2200])
        self.solar_gen['good'] = fuzz.trimf(self.solar_gen.universe, [2000, 2500, 3000])
        self.solar_gen['excellent'] = fuzz.trimf(self.solar_gen.universe, [2700, 3000, 3000])
        
        # Temperature (10-40°C) - 5 levels
        self.temperature = ctrl.Antecedent(np.arange(10, 41, 1), 'temperature')
        self.temperature['very_cold'] = fuzz.trimf(self.temperature.universe, [10, 10, 15])
        self.temperature['cold'] = fuzz.trimf(self.temperature.universe, [12, 17, 22])
        self.temperature['moderate'] = fuzz.trimf(self.temperature.universe, [20, 25, 30])
        self.temperature['hot'] = fuzz.trimf(self.temperature.universe, [28, 33, 38])
        self.temperature['very_hot'] = fuzz.trimf(self.temperature.universe, [35, 40, 40])
        
        # Time of Day (0-24) - 6 periods
        self.time_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'time_of_day')
        self.time_of_day['late_night'] = fuzz.trimf(self.time_of_day.universe, [0, 2, 4])
        self.time_of_day['early_morning'] = fuzz.trimf(self.time_of_day.universe, [4, 6, 8])
        self.time_of_day['morning'] = fuzz.trapmf(self.time_of_day.universe, [7, 9, 11, 13])
        self.time_of_day['afternoon'] = fuzz.trapmf(self.time_of_day.universe, [12, 14, 16, 18])
        self.time_of_day['evening'] = fuzz.trimf(self.time_of_day.universe, [17, 20, 23])
        self.time_of_day['night'] = fuzz.trimf(self.time_of_day.universe, [22, 23, 24])
        
        # ==================== OUTPUT VARIABLE ====================
        
        # Appliance Control (0-100) - 7 levels
        self.appliance_control = ctrl.Consequent(np.arange(0, 101, 1), 'appliance_control')
        self.appliance_control['enable_all'] = fuzz.trimf(self.appliance_control.universe, [0, 0, 15])
        self.appliance_control['enable_most'] = fuzz.trimf(self.appliance_control.universe, [10, 20, 30])
        self.appliance_control['enable_priority'] = fuzz.trimf(self.appliance_control.universe, [25, 35, 45])
        self.appliance_control['maintain'] = fuzz.trimf(self.appliance_control.universe, [40, 50, 60])
        self.appliance_control['reduce_slight'] = fuzz.trimf(self.appliance_control.universe, [55, 65, 75])
        self.appliance_control['reduce_moderate'] = fuzz.trimf(self.appliance_control.universe, [70, 80, 90])
        self.appliance_control['reduce_aggressive'] = fuzz.trimf(self.appliance_control.universe, [85, 95, 100])
        
    def _create_fuzzy_rules(self):
        """Create 22 comprehensive fuzzy rules organized by category"""
        self.rules = []
        self.rule_metadata = {}
        
        # ==================== CATEGORY 1: OPTIMAL SOLAR CONDITIONS (5 rules) ====================
        
        rule1 = ctrl.Rule(
            self.solar_gen['excellent'] & self.energy_demand['very_low'],
            self.appliance_control['enable_all'],
            label="Rule 1"
        )
        self.rules.append(rule1)
        self.rule_metadata[1] = {
            'category': 'Optimal Solar',
            'description': 'Excellent solar + very low demand → Enable all appliances',
            'rationale': 'Abundant solar energy with minimal demand allows full operation'
        }
        
        rule2 = ctrl.Rule(
            self.solar_gen['excellent'] & self.energy_demand['low'],
            self.appliance_control['enable_all'],
            label="Rule 2"
        )
        self.rules.append(rule2)
        self.rule_metadata[2] = {
            'category': 'Optimal Solar',
            'description': 'Excellent solar + low demand → Enable all appliances',
            'rationale': 'Strong solar generation exceeds low demand'
        }
        
        rule3 = ctrl.Rule(
            self.solar_gen['good'] & self.energy_demand['low'],
            self.appliance_control['enable_most'],
            label="Rule 3"
        )
        self.rules.append(rule3)
        self.rule_metadata[3] = {
            'category': 'Optimal Solar',
            'description': 'Good solar + low demand → Enable most appliances',
            'rationale': 'Good solar conditions support most operations'
        }
        
        rule4 = ctrl.Rule(
            self.solar_gen['excellent'] & self.energy_demand['medium'] & self.time_of_day['afternoon'],
            self.appliance_control['enable_most'],
            label="Rule 4"
        )
        self.rules.append(rule4)
        self.rule_metadata[4] = {
            'category': 'Optimal Solar',
            'description': 'Peak solar afternoon + medium demand → Enable most',
            'rationale': 'Afternoon solar peak allows expanded operations'
        }
        
        rule5 = ctrl.Rule(
            self.solar_gen['good'] & self.temperature['hot'] & self.time_of_day['afternoon'],
            self.appliance_control['enable_priority'],
            label="Rule 5"
        )
        self.rules.append(rule5)
        self.rule_metadata[5] = {
            'category': 'Optimal Solar',
            'description': 'Good solar + hot afternoon → Enable cooling priority',
            'rationale': 'Use solar energy for cooling during hot afternoon'
        }
        
        # ==================== CATEGORY 2: CRITICAL/EMERGENCY (4 rules) ====================
        
        rule6 = ctrl.Rule(
            self.energy_demand['very_high'] & self.solar_gen['none'],
            self.appliance_control['reduce_aggressive'],
            label="Rule 6"
        )
        self.rules.append(rule6)
        self.rule_metadata[6] = {
            'category': 'Critical/Emergency',
            'description': 'EMERGENCY - Very high demand + no solar',
            'rationale': 'Critical overload situation requires immediate action'
        }
        
        rule7 = ctrl.Rule(
            self.energy_demand['very_high'] & self.solar_gen['poor'],
            self.appliance_control['reduce_aggressive'],
            label="Rule 7"
        )
        self.rules.append(rule7)
        self.rule_metadata[7] = {
            'category': 'Critical/Emergency',
            'description': 'CRITICAL - Very high demand + poor solar',
            'rationale': 'Demand far exceeds available solar generation'
        }
        
        rule8 = ctrl.Rule(
            self.energy_demand['high'] & self.solar_gen['none'] & self.time_of_day['evening'],
            self.appliance_control['reduce_moderate'],
            label="Rule 8"
        )
        self.rules.append(rule8)
        self.rule_metadata[8] = {
            'category': 'Critical/Emergency',
            'description': 'Evening peak + no solar → Moderate reduction',
            'rationale': 'Evening peak without solar support requires load reduction'
        }
        
        rule9 = ctrl.Rule(
            self.temperature['very_hot'] & self.solar_gen['poor'] & self.energy_demand['high'],
            self.appliance_control['reduce_moderate'],
            label="Rule 9"
        )
        self.rules.append(rule9)
        self.rule_metadata[9] = {
            'category': 'Critical/Emergency',
            'description': 'Very hot + poor solar + high demand',
            'rationale': 'Extreme temperature with inadequate solar requires reduction'
        }
        
        # ==================== CATEGORY 3: TEMPERATURE-BASED (4 rules) ====================
        
        rule10 = ctrl.Rule(
            self.temperature['very_hot'] & self.solar_gen['excellent'],
            self.appliance_control['enable_priority'],
            label="Rule 10"
        )
        self.rules.append(rule10)
        self.rule_metadata[10] = {
            'category': 'Temperature-Based',
            'description': 'Very hot + excellent solar → Enable AC priority',
            'rationale': 'Utilize abundant solar for cooling during extreme heat'
        }
        
        rule11 = ctrl.Rule(
            self.temperature['hot'] & self.solar_gen['moderate'],
            self.appliance_control['enable_priority'],
            label="Rule 11"
        )
        self.rules.append(rule11)
        self.rule_metadata[11] = {
            'category': 'Temperature-Based',
            'description': 'Hot + moderate solar → Enable cooling',
            'rationale': 'Moderate solar supports necessary cooling'
        }
        
        rule12 = ctrl.Rule(
            self.temperature['very_cold'] & self.energy_demand['high'],
            self.appliance_control['reduce_slight'],
            label="Rule 12"
        )
        self.rules.append(rule12)
        self.rule_metadata[12] = {
            'category': 'Temperature-Based',
            'description': 'Very cold + high demand → Heating priority',
            'rationale': 'Prioritize heating while managing high demand'
        }
        
        rule13 = ctrl.Rule(
            self.temperature['moderate'] & self.solar_gen['moderate'] & self.energy_demand['medium'],
            self.appliance_control['maintain'],
            label="Rule 13"
        )
        self.rules.append(rule13)
        self.rule_metadata[13] = {
            'category': 'Temperature-Based',
            'description': 'All moderate conditions → Maintain balance',
            'rationale': 'Stable conditions allow balanced operation'
        }
        
        # ==================== CATEGORY 4: TIME-OF-DAY (4 rules) ====================
        
        rule14 = ctrl.Rule(
            self.time_of_day['late_night'] & self.energy_demand['low'],
            self.appliance_control['enable_all'],
            label="Rule 14"
        )
        self.rules.append(rule14)
        self.rule_metadata[14] = {
            'category': 'Time-of-Day',
            'description': 'Late night + low demand → Off-peak advantage',
            'rationale': 'Off-peak hours with low demand allow full operation'
        }
        
        rule15 = ctrl.Rule(
            self.time_of_day['early_morning'] & self.solar_gen['poor'] & self.energy_demand['medium'],
            self.appliance_control['enable_priority'],
            label="Rule 15"
        )
        self.rules.append(rule15)
        self.rule_metadata[15] = {
            'category': 'Time-of-Day',
            'description': 'Early morning → Priority only',
            'rationale': 'Limited solar at dawn requires priority focus'
        }
        
        rule16 = ctrl.Rule(
            self.time_of_day['evening'] & self.energy_demand['high'] & self.solar_gen['poor'],
            self.appliance_control['reduce_moderate'],
            label="Rule 16"
        )
        self.rules.append(rule16)
        self.rule_metadata[16] = {
            'category': 'Time-of-Day',
            'description': 'Evening peak + poor solar',
            'rationale': 'Evening peak without solar requires moderation'
        }
        
        rule17 = ctrl.Rule(
            self.time_of_day['night'] & self.energy_demand['medium'],
            self.appliance_control['enable_priority'],
            label="Rule 17"
        )
        self.rules.append(rule17)
        self.rule_metadata[17] = {
            'category': 'Time-of-Day',
            'description': 'Night + medium demand → Priority only',
            'rationale': 'Nighttime requires focused energy use'
        }
        
        # ==================== CATEGORY 5: BALANCED OPERATION (3 rules) ====================
        
        rule18 = ctrl.Rule(
            self.solar_gen['moderate'] & self.energy_demand['medium'] & self.temperature['moderate'],
            self.appliance_control['maintain'],
            label="Rule 18"
        )
        self.rules.append(rule18)
        self.rule_metadata[18] = {
            'category': 'Balanced Operation',
            'description': 'All moderate → Maintain current state',
            'rationale': 'Balanced conditions support stable operation'
        }
        
        rule19 = ctrl.Rule(
            self.solar_gen['moderate'] & self.energy_demand['low'],
            self.appliance_control['enable_most'],
            label="Rule 19"
        )
        self.rules.append(rule19)
        self.rule_metadata[19] = {
            'category': 'Balanced Operation',
            'description': 'Moderate solar + low demand → Enable most',
            'rationale': 'Good balance allows expanded operations'
        }
        
        rule20 = ctrl.Rule(
            self.energy_demand['medium'] & self.solar_gen['poor'],
            self.appliance_control['reduce_slight'],
            label="Rule 20"
        )
        self.rules.append(rule20)
        self.rule_metadata[20] = {
            'category': 'Balanced Operation',
            'description': 'Medium demand + poor solar',
            'rationale': 'Inadequate solar requires minor adjustments'
        }
        
        # ==================== CATEGORY 6: ADDITIONAL OPTIMIZATION (2 rules) ====================
        
        rule21 = ctrl.Rule(
            self.solar_gen['good'] & self.energy_demand['high'] & self.temperature['moderate'],
            self.appliance_control['enable_priority'],
            label="Rule 21"
        )
        self.rules.append(rule21)
        self.rule_metadata[21] = {
            'category': 'Additional Optimization',
            'description': 'Good solar + high demand → Priority management',
            'rationale': 'Good solar supports prioritized high demand'
        }
        
        rule22 = ctrl.Rule(
            self.time_of_day['morning'] & self.solar_gen['moderate'] & self.energy_demand['low'],
            self.appliance_control['enable_all'],
            label="Rule 22"
        )
        self.rules.append(rule22)
        self.rule_metadata[22] = {
            'category': 'Additional Optimization',
            'description': 'Morning + moderate solar + low demand',
            'rationale': 'Morning conditions favorable for full operation'
        }
        
    def compute(self, energy_demand, solar_gen, temperature, time_of_day):
        """
        Compute fuzzy control output with enhanced error handling
        
        Returns:
            float: Control output (0-100)
        """
        try:
            self.controller.reset()
            self.controller.input['energy_demand'] = float(np.clip(energy_demand, 0, 5000))
            self.controller.input['solar_gen'] = float(np.clip(solar_gen, 0, 3000))
            self.controller.input['temperature'] = float(np.clip(temperature, 10, 40))
            self.controller.input['time_of_day'] = float(time_of_day % 24)
            self.controller.compute()
            
            output = self.controller.output['appliance_control']
            
            # Log decision
            self._log_decision(energy_demand, solar_gen, temperature, time_of_day, output)
            
            return output
            
        except Exception as e:
            print(f"⚠️  Fuzzy computation error: {e}")
            return 50.0  # Default maintain value
    
    def _log_decision(self, demand, solar, temp, time, output):
        """Log decision for analysis"""
        self.decision_log.append({
            'timestamp': datetime.now().isoformat(),
            'inputs': {
                'demand': demand,
                'solar': solar,
                'temperature': temp,
                'time': time
            },
            'output': output,
            'decision': self._interpret_output(output)
        })
        
        # Keep only last 1000 decisions
        if len(self.decision_log) > 1000:
            self.decision_log.pop(0)
    
    def _interpret_output(self, output):
        """Interpret fuzzy output into human-readable decision"""
        if output < 15:
            return "Enable All Appliances"
        elif output < 30:
            return "Enable Most Appliances"
        elif output < 45:
            return "Enable Priority Appliances"
        elif output < 60:
            return "Maintain Balance"
        elif output < 75:
            return "Reduce Slightly"
        elif output < 90:
            return "Reduce Moderately"
        else:
            return "Reduce Aggressively"
    
    def get_rule_categories(self):
        """Return all rules organized by category"""
        categories = {}
        for rule_id, metadata in self.rule_metadata.items():
            category = metadata['category']
            if category not in categories:
                categories[category] = []
            categories[category].append({
                'id': rule_id,
                'description': metadata['description'],
                'rationale': metadata['rationale']
            })
        return categories
    
    def get_active_rules(self, demand, solar, temp, time_hour):
        """
        Identify which rules are currently active based on inputs
        
        Returns:
            list: Active rule information with activation levels
        """
        active_rules = []
        
        # Get membership values for all inputs
        demand_memberships = {
            'very_low': fuzz.interp_membership(self.energy_demand.universe, 
                                              self.energy_demand['very_low'].mf, demand),
            'low': fuzz.interp_membership(self.energy_demand.universe, 
                                         self.energy_demand['low'].mf, demand),
            'medium': fuzz.interp_membership(self.energy_demand.universe, 
                                            self.energy_demand['medium'].mf, demand),
            'high': fuzz.interp_membership(self.energy_demand.universe, 
                                          self.energy_demand['high'].mf, demand),
            'very_high': fuzz.interp_membership(self.energy_demand.universe, 
                                               self.energy_demand['very_high'].mf, demand)
        }
        
        solar_memberships = {
            'none': fuzz.interp_membership(self.solar_gen.universe, 
                                          self.solar_gen['none'].mf, solar),
            'poor': fuzz.interp_membership(self.solar_gen.universe, 
                                          self.solar_gen['poor'].mf, solar),
            'moderate': fuzz.interp_membership(self.solar_gen.universe, 
                                              self.solar_gen['moderate'].mf, solar),
            'good': fuzz.interp_membership(self.solar_gen.universe, 
                                          self.solar_gen['good'].mf, solar),
            'excellent': fuzz.interp_membership(self.solar_gen.universe, 
                                               self.solar_gen['excellent'].mf, solar)
        }
        
        temp_memberships = {
            'very_cold': fuzz.interp_membership(self.temperature.universe, 
                                               self.temperature['very_cold'].mf, temp),
            'cold': fuzz.interp_membership(self.temperature.universe, 
                                          self.temperature['cold'].mf, temp),
            'moderate': fuzz.interp_membership(self.temperature.universe, 
                                              self.temperature['moderate'].mf, temp),
            'hot': fuzz.interp_membership(self.temperature.universe, 
                                         self.temperature['hot'].mf, temp),
            'very_hot': fuzz.interp_membership(self.temperature.universe, 
                                              self.temperature['very_hot'].mf, temp)
        }
        
        time_memberships = {
            'late_night': fuzz.interp_membership(self.time_of_day.universe, 
                                                self.time_of_day['late_night'].mf, time_hour),
            'early_morning': fuzz.interp_membership(self.time_of_day.universe, 
                                                   self.time_of_day['early_morning'].mf, time_hour),
            'morning': fuzz.interp_membership(self.time_of_day.universe, 
                                             self.time_of_day['morning'].mf, time_hour),
            'afternoon': fuzz.interp_membership(self.time_of_day.universe, 
                                               self.time_of_day['afternoon'].mf, time_hour),
            'evening': fuzz.interp_membership(self.time_of_day.universe, 
                                             self.time_of_day['evening'].mf, time_hour),
            'night': fuzz.interp_membership(self.time_of_day.universe, 
                                           self.time_of_day['night'].mf, time_hour)
        }
        
        # Check each rule manually
        rule_conditions = {
            1: ('excellent', 'very_low', None, None),
            2: ('excellent', 'low', None, None),
            3: ('good', 'low', None, None),
            4: ('excellent', 'medium', None, 'afternoon'),
            5: ('good', None, 'hot', 'afternoon'),
            6: ('none', 'very_high', None, None),
            7: ('poor', 'very_high', None, None),
            8: ('none', 'high', None, 'evening'),
            9: ('poor', 'high', 'very_hot', None),
            10: ('excellent', None, 'very_hot', None),
            11: ('moderate', None, 'hot', None),
            12: (None, 'high', 'very_cold', None),
            13: ('moderate', 'medium', 'moderate', None),
            14: (None, 'low', None, 'late_night'),
            15: ('poor', 'medium', None, 'early_morning'),
            16: ('poor', 'high', None, 'evening'),
            17: (None, 'medium', None, 'night'),
            18: ('moderate', 'medium', 'moderate', None),
            19: ('moderate', 'low', None, None),
            20: ('poor', 'medium', None, None),
            21: ('good', 'high', 'moderate', None),
            22: ('moderate', 'low', None, 'morning')
        }
        
        # Calculate activation for each rule
        for rule_id, (solar_cond, demand_cond, temp_cond, time_cond) in rule_conditions.items():
            activation = 1.0
            
            if solar_cond:
                activation = min(activation, solar_memberships.get(solar_cond, 0))
            if demand_cond:
                activation = min(activation, demand_memberships.get(demand_cond, 0))
            if temp_cond:
                activation = min(activation, temp_memberships.get(temp_cond, 0))
            if time_cond:
                activation = min(activation, time_memberships.get(time_cond, 0))
            
            if activation > 0.1:  # Threshold for considering a rule active
                metadata = self.rule_metadata[rule_id]
                active_rules.append({
                    'rule_id': rule_id,
                    'description': metadata['description'],
                    'category': metadata['category'],
                    'activation': round(activation, 3),
                    'rationale': metadata['rationale']
                })
        
        # Sort by activation level
        active_rules.sort(key=lambda x: x['activation'], reverse=True)
        return active_rules
    
    def get_decision_explanation(self, demand, solar, temp, time_hour, output):
        """
        Generate human-readable explanation for a decision
        
        Returns:
            dict: Detailed explanation of the decision
        """
        decision = self._interpret_output(output)
        active_rules = self.get_active_rules(demand, solar, temp, time_hour)
        
        return {
            'decision': decision,
            'confidence': output,
            'primary_factors': [
                f"Energy Demand: {demand:.0f}W",
                f"Solar Generation: {solar:.0f}W",
                f"Temperature: {temp}°C",
                f"Time: {time_hour}:00"
            ],
            'active_rules': active_rules[:3],  # Top 3 rules
            'recommendation': self._get_recommendation(output, demand, solar)
        }
    
    def _get_recommendation(self, output, demand, solar):
        """Generate actionable recommendation"""
        if output < 30:
            return "Optimal conditions - all systems operating efficiently"
        elif output < 60:
            return "Balanced operation - monitor for changes"
        elif output < 80:
            return "Consider reducing non-essential appliances"
        else:
            return "Critical - immediate load reduction recommended"
    
    def export_rules(self, filepath='fuzzy_rules.json'):
        """Export rule metadata to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.rule_metadata, f, indent=2)
        print(f"✓ Rules exported to {filepath}")
    
    def print_rule_summary(self):
        """Print formatted summary of all rules"""
        categories = self.get_rule_categories()
        
        print("\n" + "=" * 80)
        print("FUZZY RULE BASE SUMMARY - 22 COMPREHENSIVE RULES")
        print("=" * 80)
        
        for category, rules in categories.items():
            print(f"\n{category} ({len(rules)} rules):")
            print("-" * 80)
            for rule in rules:
                print(f"  Rule {rule['id']}: {rule['description']}")
                print(f"           → {rule['rationale']}")
        
        print("\n" + "=" * 80)


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ENHANCED FUZZY LOGIC DECISION MODULE - DEMONSTRATION")
    print("=" * 80)
    
    # Initialize controller
    controller = EnhancedFuzzyEnergyController()
    
    # Print rule summary
    controller.print_rule_summary()
    
    # Test scenarios
    print("\n" + "=" * 80)
    print("TESTING DECISION SCENARIOS")
    print("=" * 80)
    
    test_scenarios = [
        {
            'name': 'Optimal Conditions',
            'demand': 800,
            'solar': 2800,
            'temp': 25,
            'time': 14
        },
        {
            'name': 'Critical Emergency',
            'demand': 4800,
            'solar': 50,
            'temp': 38,
            'time': 19
        },
        {
            'name': 'Evening Peak',
            'demand': 3500,
            'solar': 200,
            'temp': 28,
            'time': 20
        },
        {
            'name': 'Night Off-Peak',
            'demand': 600,
            'solar': 0,
            'temp': 22,
            'time': 2
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"📋 Scenario: {scenario['name']}")
        print(f"{'='*80}")
        print(f"Inputs:")
        print(f"  • Demand: {scenario['demand']}W")
        print(f"  • Solar: {scenario['solar']}W")
        print(f"  • Temperature: {scenario['temp']}°C")
        print(f"  • Time: {scenario['time']}:00")
        
        output = controller.compute(
            scenario['demand'],
            scenario['solar'],
            scenario['temp'],
            scenario['time']
        )
        
        explanation = controller.get_decision_explanation(
            scenario['demand'],
            scenario['solar'],
            scenario['temp'],
            scenario['time'],
            output
        )
        
        print(f"\n🎯 Decision: {explanation['decision']}")
        print(f"📊 Control Value: {output:.2f}/100")
        print(f"💡 Recommendation: {explanation['recommendation']}")
    
    print("\n" + "=" * 80)
    print("✅ DECISION MODULE READY FOR INTEGRATION")
    print("=" * 80)