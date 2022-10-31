import random
from random import randint
import json
from typing import TypeAlias

###############################################################################
# Config
###############################################################################

GENERATION_SEED           = 410

MIN_FEATURES_NUM          = 4
MAX_FEATURES_NUM          = 7

MIN_VALUES_NUM            = 2
MAX_VALUES_NUM            = 10
MIN_NORMAL_VALUES_NUM     = 1
MIN_ABNORMAL_VALUES_NUM   = 1

MIN_PERIODS_NUM           = 1
MAX_PERIODS_NUM           = 5
MIN_PERIOD_DURATION       = 1
MAX_PERIOD_DURATION       = 24
MIN_VALUES_PER_PERIOD     = 1
MAX_VALUES_PER_PERIOD     = 3

MIN_CLASSES_NUM           = 1
MAX_CLASSES_NUM           = 5
MIN_FEATURES_IN_CLASS_NUM = 2

###############################################################################
# Validation
###############################################################################

assert MIN_FEATURES_NUM <= MAX_FEATURES_NUM
assert MIN_FEATURES_NUM >= MIN_FEATURES_IN_CLASS_NUM

assert MIN_PERIODS_NUM <= MAX_PERIODS_NUM
assert MIN_PERIOD_DURATION < MAX_PERIOD_DURATION

assert MIN_VALUES_NUM <= MAX_VALUES_NUM
assert MIN_VALUES_NUM >= MIN_NORMAL_VALUES_NUM + MIN_ABNORMAL_VALUES_NUM

assert MIN_CLASSES_NUM <= MAX_CLASSES_NUM

assert MIN_VALUES_PER_PERIOD <= MAX_VALUES_PER_PERIOD
assert MIN_VALUES_NUM >= MIN_VALUES_PER_PERIOD * 2

###############################################################################
# Types definition
###############################################################################

TFeature: TypeAlias = dict[str, list[str], list[str]]
TPeriod : TypeAlias = dict[int, int, list[str]]
TClass  : TypeAlias = dict[str, list[TPeriod]]

###############################################################################
# Procedures definition
###############################################################################

def generate_features() -> list[TFeature]:
    result = []

    features_num = randint(MIN_FEATURES_NUM, MAX_FEATURES_NUM)

    for i in range(features_num):
        values_num        = randint(MIN_VALUES_NUM, MAX_VALUES_NUM)
        normal_values_num = randint(MIN_NORMAL_VALUES_NUM, values_num - MIN_ABNORMAL_VALUES_NUM)

        result.append(
            {
                'name'         : "feature_{}".format(i),
                'values'       : ["f_{}_value_{}".format(i, j) for j in range(values_num)],
                'normal_values': ["f_{}_value_{}".format(i, j) for j in range(normal_values_num)]
            }
        )

    return result



def generate_periods(feature_values: list[str]) -> list[TPeriod]:
    result = []

    periods_num = randint(MIN_PERIODS_NUM, MAX_PERIODS_NUM)

    for i in range(periods_num):
        duration_lower = randint(MIN_PERIOD_DURATION, MAX_PERIOD_DURATION - 1)
        duration_upper = randint(duration_lower, MAX_PERIOD_DURATION)

        period_values = []

        for i in range(periods_num):
            if i == 0:
                count = randint(MIN_VALUES_PER_PERIOD, min(MAX_VALUES_PER_PERIOD, len(feature_values) - MIN_VALUES_PER_PERIOD))
                period_values.append(random.sample(feature_values, count))
            else:
                possible_values = list(v for v in feature_values if not v in period_values[i - 1])
                count = randint(MIN_VALUES_PER_PERIOD, min(MAX_VALUES_PER_PERIOD, len(possible_values)))
                period_values.append(random.sample(possible_values, count))

        result.append(
            {
                'duration_lower': duration_lower,
                'duration_upper': duration_upper,
                'values'        : period_values
            }
        )

    return result



def generate_classes(features: list[TFeature]) -> list[TClass]:
    result = []

    classes_num  = randint(MIN_CLASSES_NUM, MAX_CLASSES_NUM)

    for i in range(classes_num):
        features_in_class_num = randint(MIN_FEATURES_IN_CLASS_NUM, len(features))
        selected_features = sorted(random.sample(features, features_in_class_num), key=lambda d: d['name'])

        class_symptoms = []

        for feature in selected_features:
            class_symptoms.append(
                {
                    'feature': feature['name'],
                    'periods': generate_periods(feature['values'])
                }
            )

        result.append(
            {
                'name'    : "disease_{}".format(i),
                'symptoms': class_symptoms
            }
        )

    return result

###############################################################################
# Main
###############################################################################

if __name__ == '__main__':

    random.seed(GENERATION_SEED)

    knowledge_database_model = {
        'Features': None,
        'Classes' : None
    }

    knowledge_database_model['Features'] = generate_features()
    knowledge_database_model['Classes']  = generate_classes(knowledge_database_model['Features'])

    with open('data.json', 'w') as f:
        json.dump(knowledge_database_model, f, ensure_ascii=False, indent=4)

    exit(0)