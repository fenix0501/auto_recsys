"""Feature groups of the manual set, used for drop-one-group ablations on validation."""
from ..common.candidates import COMMON_FEATURES
from .features import MANUAL_FEATURES

DEMOGRAPHICS=['u_age','u_income','u_sex','u_kids']
ITEM_STATIC=['i_content_type','i_year','i_age_rating','i_genre_count']
SEGMENTS=['i_agegroup_lift','i_gender_lift']
INDIRECT_FEATURES=[c for c in MANUAL_FEATURES if c.startswith('ui_') or c in SEGMENTS]

PRIMARY={
    'candidate_scores': list(COMMON_FEATURES),
    'user_demographics': DEMOGRAPHICS,
    'user_behavior': [c for c in MANUAL_FEATURES if c.startswith('u_') and c not in DEMOGRAPHICS],
    'item_static': ITEM_STATIC,
    'item_behavior': [c for c in MANUAL_FEATURES if c.startswith('i_') and c not in [*ITEM_STATIC,*SEGMENTS]],
    'indirect_all': INDIRECT_FEATURES,
}
INDIRECT={
    'genre_affinity': ['ui_genre_share','ui_genre_share7','ui_genre_share14','ui_genre_share30',
                       'ui_genre_max_share7','ui_genre_max_share14','ui_genre_max_share30',
                       'ui_genre_max_shareall','ui_genre_decay_share14','ui_genre_lift'],
    'genre_history': ['ui_genre_last_seen','ui_genre_completion','ui_genre_novelty'],
    'recent_context': ['ui_genre_lastday_overlap','ui_genre_last3days_overlap'],
    'content_type': ['ui_type_share','ui_type_share7','ui_type_share30'],
    'country': ['ui_country_share','ui_country_last_seen'],
    'director': ['ui_director_share','ui_director_last_seen'],
    'numeric_taste': ['ui_release_year_gap','ui_age_rating_gap'],
    'segment_popularity': SEGMENTS,
}
