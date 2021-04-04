"""A statically typed decision tree"""
from itertools import filterfalse
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
)


DataPoint = TypeVar('DataPoint')
Outcome = TypeVar('Outcome')

Metric = bool
Classifier = Callable[[DataPoint], Metric]


def _return_true(arg: Any) -> Literal[True]: # pylint: disable=unused-argument
    return True


def _majority(outcomes: Sequence[Outcome]) -> Outcome:
    """Return the outcome that is most common in the given sequence"""

    counts: dict[Outcome, int] = dict()
    for outcome in outcomes:
        counts[outcome] = counts.get(outcome, 0) + 1
    return max(counts, key=lambda key: counts[key])


class Model(Generic[DataPoint, Outcome]):
    """A trained decision tree model"""

    def __init__(
        self,
        middle: Optional[Outcome],
        classifier: Classifier = _return_true,
        left: Optional[Model] = None,
        right: Optional[Model] = None,
    ):
        self.classifier = classifier
        self.middle = middle
        self.left = left
        self.right = right


    def decide(self, datapoint: DataPoint) -> Outcome:
        """Predict an Outcome for the given datapoint"""

        left = self.classifier(datapoint)
        if left and self.left is not None:
            return self.left.decide(datapoint)
        if not left and self.right is not None:
            return self.right.decide(datapoint)
        if self.middle is not None:
            return self.middle
        raise Exception('Undecidable')


    def __repr__(self) -> str:
        return f'Model(classifier={self.classifier}, middle={self.middle}, ' \
                + 'left={self.left}, right={self.right})'

    # TODO pickling to disk


class DecisionTree(Generic[DataPoint, Outcome]):
    """A decision tree"""

    classifiers: Sequence[Classifier]
    training_set: list[tuple[DataPoint, Outcome]]
    model: Model


    def __init__(self, classifiers: Sequence[Classifier]):
        self.classifiers = classifiers
        self.training_set = list()
        self.model = Model(None)


    def reset(self) -> None:
        """Clear the training set and model"""
        self.training_set = list()
        self.model = Model(None)


    def train(self) -> None:
        """Build the model"""
        self.model = self._train(self.classifiers, self.training_set)


    def _train(
        self,
        classifiers: Sequence[Classifier],
        data: Sequence[tuple[DataPoint, Outcome]]
    ) -> Model:
        """Create a model from the given classifiers and training data"""
        outcomes = list(map(lambda d: d[1], data))

        # If the data is empty this is a bad leaf
        if len(data) == 0:
            return Model(None)

        # If classifiers is empty, return the most likely from data
        if len(classifiers) == 0:
            return Model(_majority(outcomes))

        # If all the data has the same outcome, return Model(outcome)
        if len(set(outcomes)) == 1:
            return Model(outcomes[0])

        # Else:
        # Pick the best classifier TODO
#       def score(
#           left_asdf: Iterable[tuple[DataPoint, Outcome]],
#           right_asdf: Iterable[tuple[DataPoint, Outcome]],
#       ):
#            left = list(map(lambda d: d[1], left_asdf))
#            right = list(map(lambda d: d[1], right_asdf))
#            scores = tuple(map(lambda o: abs(left.count(o) - right.count(o)), outcomes))
#
#        foo = [
#                (
#                    classifier,
#                    list(filter(lambda d: classifier(d[0]), data)),
#                    list(filter(lambda d: not classifier(d[0]), data)),
#                )
#                for classifier in classifiers
#            ]
        # Make a new model
        return Model(
            None,
            classifier=classifiers[0],
            left=self._train(classifiers[1:],
                list(filter(lambda d: classifiers[0](d[0]), data))
            ),
            right=self._train(classifiers[1:],
                list(filterfalse(lambda d: classifiers[0](d[0]), data))
            ),
        )


    def add_training_data(self, data: Iterable[tuple[DataPoint, Outcome]]) -> None:
        """Add training data to the tree"""
        self.training_set.extend(data)


    def decide(self, datapoint: DataPoint) -> Outcome:
        """Get the decision tree's prediction for the given datapoint"""
        return self.model.decide(datapoint)
