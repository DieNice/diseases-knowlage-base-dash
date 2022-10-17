import json
from functools import partial
from random import randint, sample
from typing import Dict, List, TypeAlias

from app import app
from dash_extensions.enrich import Input, Output, State

TFeature: TypeAlias = Dict[str, List[str]]
TPeriod: TypeAlias = Dict[int, List[str]]
TClass: TypeAlias = Dict[str, List[TPeriod]]


def generate_features(min_features: int,
                      max_features: int,
                      min_values: int,
                      max_values: int,
                      ) -> list[TFeature]:
    """Generation features for classes by parameters

    Args:
        min_features (int): min num features
        max_features (int): max num features
        min_values (int): min value for feature
        max_values (int): max value for feature

    Returns:
        list[TFeature]: List of features
    """
    MIN_NORMAL_VALUES_NUM = 1
    MIN_ABNORMAL_VALUES_NUM = 1

    result = []

    features_num = randint(min_features, max_features)

    for i in range(features_num):
        values_num = randint(min_values, max_values)
        normal_values_num = randint(
            MIN_NORMAL_VALUES_NUM, values_num - MIN_ABNORMAL_VALUES_NUM)

        result.append(
            {
                'name': "feature_{}".format(i),
                'values': ["f_{}_value_{}".format(i, j) for j in range(values_num)],
                'normal_values': ["f_{}_value_{}".format(i, j) for j in range(normal_values_num)]
            }
        )

    return result


def generate_periods(min_periods_num: int,
                     max_periods_num: int,
                     min_periods_duration: int,
                     max_periods_duration: int,
                     min_value_by_period: int,
                     max_value_by_period: int,
                     feature_values: list[str],
                     ) -> list[TPeriod]:
    """Generation periods

    Args:
        feature_values (list[str]): List of features
        min_periods_num (int): Min period num
        max_periods_num (int): Max period num
        min_periods_duration (int): Min period duration
        max_periods_duration (int): Max period duration
        min_value_by_period (int): Min value by period
        max_value_by_period (int): Max value by period

    Returns:
        list[TPeriod]: List of periods
    """
    result = []

    periods_num = randint(min_periods_num, max_periods_num)

    for i in range(periods_num):
        duration_lower = randint(min_periods_duration,
                                 max_periods_duration - 1)
        duration_upper = randint(duration_lower, max_periods_duration)

        period_values = []

        for i in range(periods_num):
            if i == 0:
                value_1 = abs(min_value_by_period)
                value_2 = abs(min(max_value_by_period, len(
                    feature_values) - min_value_by_period))

                count = randint(min(value_1, value_2), max(
                    value_1, value_2)) % len(feature_values)
                period_values.append(sample(feature_values, count))
            else:
                possible_values = list(
                    v for v in feature_values if not v in period_values[i - 1])

                value_1 = abs(min_value_by_period)
                value_2 = abs(min(
                    max_value_by_period, len(possible_values)))

                count = randint(min(value_1, value_2), max(
                    value_1, value_2)) % len(possible_values)

                period_values.append(sample(possible_values, count))

        result.append(
            {
                'duration_lower': duration_lower,
                'duration_upper': duration_upper,
                'values': period_values
            }
        )

    return result


def generate_classes(features: list[TFeature],
                     min_classes_num: int,
                     max_classes_num: int,
                     min_features_in_class: int,
                     class_name_pattern: str,
                     generate_periods_partial: partial) -> list[TClass]:
    result = []

    classes_num = randint(min_classes_num, max_classes_num)

    for i in range(classes_num):
        min_v = min(min_features_in_class, len(features))
        max_v = max(min_features_in_class, len(features))
        features_in_class_num = randint(min_v, max_v) % len(features)

        selected_features = sorted(sample(
            features, features_in_class_num), key=lambda d: d['name'])

        class_symptoms = []

        for feature in selected_features:
            class_symptoms.append(
                {
                    'feature': feature['name'],
                    'periods': generate_periods_partial(feature['values'])
                }
            )

        result.append(
            {
                'name': f"{class_name_pattern}_{i}",
                'symptoms': class_symptoms
            }
        )

    return result


@app.callback(
    output={
        "alert": Output("alert-id", "children"),
        "download": Output("download-classes-id", "data")
    },
    inputs={
        "generate": Input("generate-btn-id", "n_clicks"),
        "seed": State("generation-seed-id", "value"),
        "min_features_num": State("min-features-num-id", "value"),
        "max_features_num": State("max-features-num-id", "value"),
        "min_values_num": State("min-values-num-id", "value"),
        "max_values_num": State("max-values-num-id", "value"),
        "min_periods_num": State("min-periods-num-id", "value"),
        "max_periods_num": State("max-periods-num-id", "value"),
        "min_period_duration": State("min-period-duration-id", "value"),
        "max_period_duration": State("max-period-duration-id", "value"),
        "min_values_by_period": State("min-values-by-period-id", "value"),
        "max_values_by_period": State("max-values-by-period-id", "value"),
        "min_classes_num": State("min-classes-num-id", "value"),
        "max_classes_num": State("max-classes-num-id", "value"),
        "min_features_in_classes": State("min-features-in-class-id", "value"),
        "max_features_in_classes": State("max-features-in-class-id", "value"),
        "name_class_pattern": State("name-class-pattern-id", "value"),
    },
    prevent_initial_call=True
)
def generate(generate: int, seed: int,
             min_features_num: int,
             max_features_num: int,
             min_values_num: int,
             max_values_num: int,
             min_periods_num: int,
             max_periods_num: int,
             min_period_duration: int,
             max_period_duration: int,
             min_values_by_period: int,
             max_values_by_period: int,
             min_classes_num: int,
             max_classes_num: int,
             min_features_in_classes: int,
             max_features_in_classes: int,
             name_class_pattern: str,
             ):

    knowledge_database_model = {
        'Classes': None
    }

    features = generate_features(
        min_features_num, max_features_num, min_values_num, max_values_num)

    generate_periods_func = partial(generate_periods, min_periods_num, max_periods_num,
                                    min_period_duration, max_period_duration, min_values_by_period, max_values_by_period)

    knowledge_database_model['Classes'] = generate_classes(features, min_classes_num,
                                                           max_classes_num, min_features_in_classes,
                                                           name_class_pattern, generate_periods_func)

    data_file = json.dumps(knowledge_database_model)
    result_file = {"content": data_file,
                   "filename": f"{name_class_pattern}_generation.json"}

    return {"alert": "Generation successfull",
            "download": result_file}
