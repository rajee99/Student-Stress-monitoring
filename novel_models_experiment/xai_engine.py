"""
Explainable AI (XAI) & Prescriptive Recommendation Engine for Non-Tree Models.

Features:
1. Multi-Dimensional Stress Factor Decomposition:
   - Academic Stress
   - Psychological Stress
   - Physical Stress
   - Environmental Stress
   - Social Stress
2. Student-Level Dimension Contribution & Trigger Isolation
3. Actionable Prescriptive Recommendation Rules
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


# 1. Stress Dimension Mapping across Survey Features
STRESS_DIMENSIONS = {
    "Academic Stress": [
        "academic_performance",
        "study_load",
        "teacher_student_relationship",
        "future_career_concerns",
        "academic_pressure_ratio"
    ],
    "Psychological Stress": [
        "anxiety_level",
        "depression",
        "self_esteem",
        "mental_health_history",
        "mental_strain_composite"
    ],
    "Physical Stress": [
        "headache",
        "blood_pressure",
        "sleep_quality",
        "breathing_problem",
        "physiological_load_index"
    ],
    "Environmental Stress": [
        "noise_level",
        "living_conditions",
        "safety",
        "basic_needs"
    ],
    "Social Stress": [
        "social_support",
        "peer_pressure",
        "extracurricular_activities",
        "bullying"
    ]
}

# 2. Actionable Recommendation Rules by Dimension & Severity
RECOMMENDATION_RULES = {
    "Academic Stress": {
        "Low": [
            "Maintain current positive academic study habits",
            "Optional peer tutoring participation"
        ],
        "Medium": [
            "Weekly Study Planning",
            "Time Management Coaching",
            "Academic Advisor Meeting"
        ],
        "High": [
            "Study Load Reduction Plan",
            "Time Management Coaching",
            "Academic Advisor Meeting",
            "Career Counseling Support"
        ]
    },
    "Psychological Stress": {
        "Low": [
            "Maintain positive mental wellness and self-care routines"
        ],
        "Medium": [
            "Stress Management Workshops",
            "Mindfulness & Meditation Training",
            "Peer Support Group Engagement"
        ],
        "High": [
            "Confidential Counseling Services Consultation",
            "Mental Health Specialist Consultation",
            "Stress Reduction Program",
            "Immediate Wellness Support"
        ]
    },
    "Physical Stress": {
        "Low": [
            "Continue healthy sleep and regular physical exercise"
        ],
        "Medium": [
            "Sleep Hygiene Guidance Protocol",
            "Physical Wellness & Activity Plan",
            "Ergonomic Study Habit Adjustments"
        ],
        "High": [
            "Campus Health Center Medical Checkup",
            "Structured Sleep Recovery Protocol",
            "Cardiovascular & Breathing Relaxation Therapy"
        ]
    },
    "Environmental Stress": {
        "Low": [
            "Current living and study environment is optimal"
        ],
        "Medium": [
            "Quiet Study Space Allocation",
            "Dormitory / Living Condition Review",
            "Campus Safe Space Utilization"
        ],
        "High": [
            "Student Affairs Housing Assistance",
            "Emergency Basic Needs Resource Access",
            "Campus Security Safety Support"
        ]
    },
    "Social Stress": {
        "Low": [
            "Healthy social network and peer interactions"
        ],
        "Medium": [
            "Social Club & Community Engagement",
            "Conflict Resolution Coaching",
            "Peer Mentorship Program"
        ],
        "High": [
            "Anti-Bullying Incident Intervention",
            "Dedicated Social Support Counseling",
            "Safe Social Reintegration Plan"
        ]
    }
}


class StudentXAIEngine:
    """Computes Dimension Attribution and Prescriptive Interventions for Non-Tree Models."""

    @staticmethod
    def explain_student_prediction(
        student_features: pd.Series,
        predicted_level: int,
        model_feature_weights: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Decomposes student stress risk into the 5 core dimensions using model metric weights.
        """
        level_map = {0: "Low", 1: "Medium", 2: "High"}
        severity = level_map.get(int(predicted_level), "Medium")
        
        # Align features and metric weights
        weights_dict = {f: w for f, w in zip(feature_names, model_feature_weights)}
        
        dimension_scores = {}
        dimension_feature_breakdown = {}

        for dim_name, dim_features in STRESS_DIMENSIONS.items():
            dim_total_score = 0.0
            breakdown = []
            
            for feat in dim_features:
                if feat in student_features:
                    val = float(student_features[feat])
                    weight = float(weights_dict.get(feat, 1.0 / len(feature_names)))
                    
                    # For reverse-coded indicators (like self_esteem, sleep_quality, safety, basic_needs, academic_performance)
                    # Higher raw value means lower stress in some features. Standardize impact:
                    if feat in ['self_esteem', 'sleep_quality', 'safety', 'basic_needs', 'social_support', 'teacher_student_relationship']:
                        risk_val = max(0.0, 5.0 - val)
                    else:
                        risk_val = val
                    
                    impact = risk_val * weight
                    dim_total_score += impact
                    breakdown.append({'feature': feat, 'raw_value': val, 'impact': impact})
                    
            dimension_scores[dim_name] = dim_total_score
            dimension_feature_breakdown[dim_name] = sorted(breakdown, key=lambda x: x['impact'], reverse=True)

        total_score_sum = sum(dimension_scores.values()) + 1e-8
        dimension_percentages = {
            dim: (score / total_score_sum) * 100.0 for dim, score in dimension_scores.items()
        }
        
        dominant_dimension = max(dimension_scores, key=dimension_scores.get)
        recommendations = RECOMMENDATION_RULES.get(dominant_dimension, {}).get(severity, [])
        
        return {
            'predicted_level': int(predicted_level),
            'severity': severity,
            'dominant_dimension': dominant_dimension,
            'dimension_scores': dimension_scores,
            'dimension_percentages': dimension_percentages,
            'top_factors_in_dominant': dimension_feature_breakdown[dominant_dimension][:3],
            'recommendations': recommendations
        }

    @staticmethod
    def format_student_xai_report(student_id: Any, explanation: Dict[str, Any]) -> str:
        """Generates a clean terminal and markdown summary for an individual student."""
        lines = [
            "=" * 80,
            f"STUDENT XAI STRESS REPORT: ID {student_id}",
            "=" * 80,
            f"Predicted Stress Level:      {explanation['severity']} Stress (Class {explanation['predicted_level']})",
            f"Dominant Stress Dimension:   {explanation['dominant_dimension']} ({explanation['dimension_percentages'][explanation['dominant_dimension']]:.1f}% contribution)",
            "-" * 80,
            "Stress Dimension Breakdown:"
        ]
        for dim, pct in explanation['dimension_percentages'].items():
            bar = "#" * int(pct / 4)
            lines.append(f"  - {dim:<24}: {pct:>5.1f}% | {bar}")
            
        lines.append("-" * 80)
        lines.append(f"Top Stress Triggers in {explanation['dominant_dimension']}:")
        for idx, item in enumerate(explanation['top_factors_in_dominant'], 1):
            lines.append(f"  {idx}. {item['feature']} (Value: {item['raw_value']:.1f}, Contribution Score: {item['impact']:.3f})")
            
        lines.append("-" * 80)
        lines.append("Prescriptive Actionable Recommendations:")
        for rec in explanation['recommendations']:
            lines.append(f"  -> [RECOMMENDED] {rec}")
        lines.append("=" * 80)
        return "\n".join(lines)
