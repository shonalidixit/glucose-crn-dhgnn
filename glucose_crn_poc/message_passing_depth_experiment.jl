"""
    message_passing_depth_experiment.jl

Train and evaluate directed hypergraph neural networks for reaction-level
regression on the glucose chemical reaction network.

The script contains two related experiments. The first compares one, two, and
three directed message-passing layers on the same train, validation, and test
split. The second constructs a learning curve for the one-layer model using
nested subsets of a fixed training pool.

The learning-curve evaluation reports mean signed error and mean absolute error
in the original reaction-energy units. Mean signed error is defined as
`prediction - target`, so positive values indicate average overprediction and
negative values indicate average underprediction.

Model training uses mean squared error in standardised target space because a
signed error objective can cancel positive and negative residuals. Mean signed
error is therefore used as a post hoc bias diagnostic rather than as the
optimisation objective.

For the learning curve, target standardisation is calculated once from the full
training split and reused at every training-set size. Every fitted model is then
evaluated on the same full training pool, validation set, and test set. Trained
model snapshots are saved so that later analyses can be run without retraining.
"""

using Random
using Statistics
using LinearAlgebra
using Lux
using Optimisers
using Enzyme
using MLUtils
using Serialization
using HyperGraphNeuralNetworks
using Plots

include("directed_crn.jl")


const RANDOM_SEED = 1234
const HIDDEN_DIM = 64
const NUMBER_OF_EPOCHS = 100
const LEARNING_RATE = 1.0f-3
const WEIGHT_DECAY = 1.0f-5

const TRAIN_FRACTION = 0.60
const VALIDATION_FRACTION = 0.20

"""
Percentages of the available training split used to construct the learning
curve.
"""
const LEARNING_CURVE_PERCENTAGES =
    [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]


"""
Energy unit used for the glucose reaction-energy targets.
"""
const ENERGY_UNIT = "kcal/mol"


"""
    GlucoseHyperedgeRegressor

Directed hypergraph neural network for reaction-level regression with a
configurable number of message-passing layers.

Each layer performs one directed message-passing step. The first layer maps
the original molecular fingerprints to the hidden dimension. Later layers
take the updated vertex representations from the previous layer and perform
another directed message-passing step. The final hyperedge representations
are passed to a scalar regression head.
"""
struct GlucoseHyperedgeRegressor{L,IW,IB} <: Lux.AbstractLuxLayer
    directed_layers::L
    hidden_dim::Int
    num_layers::Int
    init_weight::IW
    init_bias::IB
end


"""
    GlucoseHyperedgeRegressor(input_dim, hidden_dim; num_layers=1)

Construct a directed hypergraph regression model with 1, 2, or more
message-passing layers.

The first layer receives the original vertex features. Every later layer
receives the updated vertex features from the previous layer. No explicit
initial hyperedge features are required.
"""
function GlucoseHyperedgeRegressor(
    input_dim::Int,
    hidden_dim::Int;
    num_layers::Int = 1,
    init_weight = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
)
    input_dim > 0 ||
        throw(
            ArgumentError(
                "`input_dim` must be positive."
            )
        )

    hidden_dim > 0 ||
        throw(
            ArgumentError(
                "`hidden_dim` must be positive."
            )
        )

    num_layers > 0 ||
        throw(
            ArgumentError(
                "`num_layers` must be positive."
            )
        )

    layers = [
        DirectedHypergraphLayer(
            layer_index == 1 ? input_dim : hidden_dim,
            0,
            hidden_dim;
            activation = tanh,
            normalize = true,
        )
        for layer_index in 1:num_layers
    ]

    directed_layers =
        Tuple(layers)

    return GlucoseHyperedgeRegressor(
        directed_layers,
        hidden_dim,
        num_layers,
        init_weight,
        init_bias,
    )
end


"""
    Lux.initialparameters(rng, model)

Initialise all directed message-passing layers and the scalar regression head.
"""
function Lux.initialparameters(
    rng::AbstractRNG,
    model::GlucoseHyperedgeRegressor,
)
    directed_parameters =
        Tuple([
            Lux.initialparameters(
                rng,
                layer,
            )
            for layer in model.directed_layers
        ])

    W_output =
        permutedims(
            model.init_weight(
                rng,
                1,
                model.hidden_dim,
            )
        )

    b_output =
        permutedims(
            model.init_bias(
                rng,
                1,
                1,
            )
        )

    return (
        directed_layers = directed_parameters,
        W_output = W_output,
        b_output = b_output,
    )
end


"""
    Lux.initialstates(rng, model)

Initialise state for every directed message-passing layer.
"""
function Lux.initialstates(
    rng::AbstractRNG,
    model::GlucoseHyperedgeRegressor,
)
    directed_states =
        Tuple([
            Lux.initialstates(
                rng,
                layer,
            )
            for layer in model.directed_layers
        ])

    return (
        directed_layers = directed_states,
    )
end


"""
    Lux.parameterlength(model::GlucoseHyperedgeRegressor)

Return the total number of trainable parameters in the directed layers and
scalar regression head.

The output layer contains `hidden_dim` weights, one for each hidden hyperedge
feature, plus one scalar bias parameter.
"""
function Lux.parameterlength(
    model::GlucoseHyperedgeRegressor,
)
    directed_parameters =
        sum(
            Lux.parameterlength(layer)
            for layer in model.directed_layers
        )

    output_parameters =
        model.hidden_dim + 1

    return directed_parameters +
           output_parameters
end


"""
    Lux.statelength(model::GlucoseHyperedgeRegressor)

Return the total number of state entries used by the directed message-passing
layers.
"""
function Lux.statelength(
    model::GlucoseHyperedgeRegressor,
)
    return sum(
        Lux.statelength(layer)
        for layer in model.directed_layers
    )
end


"""
    regression_forward(model, input, ps, st)

Run the configured number of directed message-passing layers and predict one
scalar property for every reaction hyperedge.
"""
function regression_forward(
    model::GlucoseHyperedgeRegressor,
    input,
    ps,
    st,
)
    X,
    incidence_tail,
    incidence_head = input

    current_vertices =
        X

    hidden_hyperedges =
        nothing

    new_layer_states =
        Vector{Any}(
            undef,
            model.num_layers,
        )

    for layer_index in 1:model.num_layers
        directed_output,
        new_layer_state =
            model.directed_layers[layer_index](
                (
                    current_vertices,
                    incidence_tail,
                    incidence_head,
                ),
                ps.directed_layers[layer_index],
                st.directed_layers[layer_index],
            )

        current_vertices =
            directed_output.updated_vertices

        hidden_hyperedges =
            directed_output.updated_hyperedges

        new_layer_states[layer_index] =
            new_layer_state
    end

    predictions =
        hidden_hyperedges *
        ps.W_output .+
        ps.b_output

    output = (
        predictions = vec(predictions),
        hidden_hyperedges = hidden_hyperedges,
        hidden_vertices = current_vertices,
    )

    new_state = (
        directed_layers =
            Tuple(new_layer_states),
    )

    return output, new_state
end


"""
    random_regression_split(n; train_fraction, validation_fraction, seed)

Create reproducible train, validation and test splits over reaction
hyperedges using `MLUtils.splitobs`.

Integer split sizes are calculated first so the 60/20/20 convention retains
the same reaction counts across repeated runs.
"""
function random_regression_split(
    n::Int;
    train_fraction = TRAIN_FRACTION,
    validation_fraction = VALIDATION_FRACTION,
    seed = RANDOM_SEED,
)
    n > 0 ||
        throw(
            ArgumentError(
                "`n` must be positive."
            )
        )

    train_fraction > 0 ||
        throw(
            ArgumentError(
                "`train_fraction` must be positive."
            )
        )

    validation_fraction > 0 ||
        throw(
            ArgumentError(
                "`validation_fraction` must be positive."
            )
        )

    train_fraction + validation_fraction < 1 ||
        throw(
            ArgumentError(
                "Training and validation fractions must sum to less than one."
            )
        )

    n_train =
        floor(
            Int,
            train_fraction * n,
        )

    n_validation =
        floor(
            Int,
            validation_fraction * n,
        )

    rng =
        MersenneTwister(seed)

    train_indices,
    validation_indices,
    test_indices =
        MLUtils.splitobs(
            rng,
            collect(1:n);
            at = (
                n_train,
                n_validation,
            ),
            shuffle = true,
        )

    return (
        train = collect(train_indices),
        validation = collect(validation_indices),
        test = collect(test_indices),
    )
end


"""
    standardise_target(target, train_indices)

Standardise the reaction-property target using statistics calculated only
from the training set.

Using only training statistics prevents validation and test targets from
affecting preprocessing.
"""
function standardise_target(
    target,
    train_indices,
)
    training_mean =
        mean(
            target[train_indices]
        )

    training_std =
        std(
            target[train_indices]
        )

    training_std > 0 ||
        error(
            "Training target has zero variance."
        )

    standardised =
        Float32.(
            (target .- training_mean) ./
            training_std
        )

    return (
        values = standardised,
        mean = Float32(training_mean),
        std = Float32(training_std),
    )
end


"""
    regression_loss(predictions, targets, indices)

Calculate mean squared error for a selected set of reaction hyperedges.

MSE is used as the optimisation loss because squaring prevents positive and
negative residuals from cancelling. Mean signed error is reported separately
as a post hoc measure of systematic overprediction or underprediction.
"""
function regression_loss(
    predictions,
    targets,
    indices,
)
    isempty(indices) &&
        throw(
            ArgumentError(
                "Cannot calculate loss for an empty index set."
            )
        )

    residuals =
        predictions[indices] .-
        targets[indices]

    return mean(
        abs2,
        residuals,
    )
end


"""
    regression_metrics(predictions, targets, indices)

Calculate regression metrics for a selected set of reaction hyperedges.

Mean signed error is defined as `prediction - target`. Positive values indicate
average overprediction and negative values indicate average underprediction.
MAE, RMSE, and mean signed error are reported in the original target scale when
this function is called through `evaluate_regression_model`.

# Arguments

- `predictions`: Predicted reaction-property values.
- `targets`: Observed reaction-property values.
- `indices`: Reaction indices included in the calculation.

# Returns

A named tuple containing mean signed error, MAE, RMSE, and R2.
"""
function regression_metrics(
    predictions,
    targets,
    indices,
)
    isempty(indices) &&
        throw(
            ArgumentError(
                "Cannot calculate metrics for an empty index set."
            )
        )

    observed =
        targets[indices]

    predicted =
        predictions[indices]

    residuals =
        predicted .-
        observed

    mean_signed_error =
        mean(residuals)

    mae =
        mean(
            abs.(residuals)
        )

    mse =
        mean(
            abs2,
            residuals,
        )

    rmse =
        sqrt(
            mse
        )

    target_mean =
        mean(observed)

    total_sum_squares =
        sum(
            abs2,
            observed .- target_mean,
        )

    residual_sum_squares =
        sum(
            abs2,
            residuals,
        )

    r2 =
        if total_sum_squares == 0
            NaN
        else
            1 -
            residual_sum_squares /
            total_sum_squares
        end

    return (
        mean_signed_error = mean_signed_error,
        mae = mae,
        rmse = rmse,
        r2 = r2,
    )
end


"""
    zero_parameter_tree(x)

Create a zero-filled tree with the same array structure as a model parameter
tree. Enzyme writes gradients into this matching structure.
"""
zero_parameter_tree(
    x::AbstractArray
) =
    zeros(
        eltype(x),
        size(x),
    )


zero_parameter_tree(
    x::NamedTuple
) =
    NamedTuple{keys(x)}(
        map(
            zero_parameter_tree,
            values(x),
        )
    )


zero_parameter_tree(
    x::Tuple
) =
    map(
        zero_parameter_tree,
        x,
    )


zero_parameter_tree(x) = x


"""
    parameter_squared_norm(x)

Return the sum of squared entries across a nested parameter or gradient tree.
"""
parameter_squared_norm(
    x::AbstractArray
) =
    sum(
        abs2,
        x,
    )


parameter_squared_norm(
    x::NamedTuple
) =
    sum(
        parameter_squared_norm(value)
        for value in values(x)
    )


parameter_squared_norm(
    x::Tuple
) =
    sum(
        parameter_squared_norm(value)
        for value in x
    )


parameter_squared_norm(x) = 0.0


"""
    parameter_tree_finite(x)

Return `true` when every array entry in a nested parameter tree is finite.
"""
parameter_tree_finite(
    x::AbstractArray
) =
    all(
        isfinite,
        x,
    )


parameter_tree_finite(
    x::NamedTuple
) =
    all(
        parameter_tree_finite(value)
        for value in values(x)
    )


parameter_tree_finite(
    x::Tuple
) =
    all(
        parameter_tree_finite(value)
        for value in x
    )


parameter_tree_finite(x) = true


"""
    regression_objective(...)

Calculate training MSE together with L2 regularisation on the regression
head.
"""
function regression_objective(
    model,
    ps,
    st,
    X,
    incidence_tail,
    incidence_head,
    targets,
    train_indices,
    weight_decay,
)
    output, _ =
        regression_forward(
            model,
            (
                X,
                incidence_tail,
                incidence_head,
            ),
            ps,
            st,
        )

    data_loss =
        regression_loss(
            output.predictions,
            targets,
            train_indices,
        )

    regularisation =
        sum(
            abs2,
            ps.W_output,
        )

    return data_loss +
           weight_decay *
           regularisation
end


"""
    compute_regression_gradients(...)

Calculate model gradients with Enzyme reverse-mode automatic
differentiation.
"""
function compute_regression_gradients(
    model,
    ps,
    st,
    X,
    incidence_tail,
    incidence_head,
    targets,
    train_indices,
    weight_decay,
)
    gradient_parameters =
        zero_parameter_tree(ps)

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(
            Enzyme.Reverse
        ),
        regression_objective,
        Enzyme.Const(model),
        Enzyme.Duplicated(
            ps,
            gradient_parameters,
        ),
        Enzyme.Const(st),
        Enzyme.Const(X),
        Enzyme.Const(
            incidence_tail
        ),
        Enzyme.Const(
            incidence_head
        ),
        Enzyme.Const(targets),
        Enzyme.Const(
            train_indices
        ),
        Enzyme.Const(
            weight_decay
        ),
    )

    return gradient_parameters
end


"""
    gradient_norm(gradients)

Calculate the Euclidean norm of the complete parameter-gradient tree.
"""
function gradient_norm(
    gradients,
)
    return sqrt(
        parameter_squared_norm(
            gradients
        )
    )
end


"""
    evaluate_regression_model(...)

Evaluate predictions for a selected reaction split.

Loss is calculated in standardised target space. MAE, RMSE and R2 are
reported in the original target scale.
"""
function evaluate_regression_model(
    model,
    ps,
    st,
    X,
    incidence_tail,
    incidence_head,
    standardised_targets,
    original_targets,
    indices,
    target_mean,
    target_std,
)
    output,
    new_state =
        regression_forward(
            model,
            (
                X,
                incidence_tail,
                incidence_head,
            ),
            ps,
            st,
        )

    standardised_predictions =
        output.predictions

    original_predictions =
        standardised_predictions .*
        target_std .+
        target_mean

    loss =
        regression_loss(
            standardised_predictions,
            standardised_targets,
            indices,
        )

    metrics =
        regression_metrics(
            original_predictions,
            original_targets,
            indices,
        )

    return (
        loss = loss,
        metrics = metrics,
        predictions = original_predictions,
        state = new_state,
    )
end


"""
    train_regression_model(...)

Train the reaction-level directed hypergraph regression model using Adam.

The parameter set with the lowest validation loss is retained.
"""
function train_regression_model(
    model,
    ps,
    st,
    X,
    incidence_tail,
    incidence_head,
    standardised_targets,
    original_targets,
    train_indices,
    validation_indices,
    target_mean,
    target_std;
    epochs = NUMBER_OF_EPOCHS,
    learning_rate = LEARNING_RATE,
    weight_decay = WEIGHT_DECAY,
)
    epochs > 0 ||
        throw(
            ArgumentError(
                "`epochs` must be positive."
            )
        )

    learning_rate > 0 ||
        throw(
            ArgumentError(
                "`learning_rate` must be positive."
            )
        )

    weight_decay >= 0 ||
        throw(
            ArgumentError(
                "`weight_decay` cannot be negative."
            )
        )

    optimiser =
        Optimisers.Adam(
            learning_rate
        )

    optimiser_state =
        Optimisers.setup(
            optimiser,
            ps,
        )

    current_ps = ps
    current_st = st

    best_ps =
        deepcopy(current_ps)

    best_st =
        deepcopy(current_st)

    best_validation_loss =
        Inf

    best_epoch =
        0

    println()
    println("Training configuration")
    println("Epochs: ", epochs)
    println(
        "Learning rate: ",
        learning_rate,
    )
    println(
        "Weight decay: ",
        weight_decay,
    )
    println("Optimiser: Adam")
    println(
        "Automatic differentiation: Enzyme"
    )

    println()
    println("Beginning model training...")

    for epoch in 1:epochs
        gradients =
            compute_regression_gradients(
                model,
                current_ps,
                current_st,
                X,
                incidence_tail,
                incidence_head,
                standardised_targets,
                train_indices,
                weight_decay,
            )

        parameter_tree_finite(
            gradients
        ) || error(
            "Non-finite gradient detected at epoch $epoch."
        )

        current_gradient_norm =
            gradient_norm(
                gradients
            )

        optimiser_state,
        current_ps =
            Optimisers.update(
                optimiser_state,
                current_ps,
                gradients,
            )

        train_evaluation =
            evaluate_regression_model(
                model,
                current_ps,
                current_st,
                X,
                incidence_tail,
                incidence_head,
                standardised_targets,
                original_targets,
                train_indices,
                target_mean,
                target_std,
            )

        current_st =
            train_evaluation.state

        validation_evaluation =
            evaluate_regression_model(
                model,
                current_ps,
                current_st,
                X,
                incidence_tail,
                incidence_head,
                standardised_targets,
                original_targets,
                validation_indices,
                target_mean,
                target_std,
            )

        current_st =
            validation_evaluation.state

        validation_loss =
            Float64(
                validation_evaluation.loss
            )

        if validation_loss <
           best_validation_loss

            best_validation_loss =
                validation_loss

            best_epoch =
                epoch

            best_ps =
                deepcopy(current_ps)

            best_st =
                deepcopy(current_st)
        end

        if epoch == 1 ||
           epoch % 10 == 0 ||
           epoch == epochs

            println(
                "Epoch ",
                lpad(
                    string(epoch),
                    length(
                        string(epochs)
                    ),
                ),
                "/",
                epochs,
                " | train MSE = ",
                round(
                    train_evaluation.loss;
                    digits = 4,
                ),
                " | val MSE = ",
                round(
                    validation_evaluation.loss;
                    digits = 4,
                ),
                " | val MAE = ",
                round(
                    validation_evaluation.metrics.mae;
                    digits = 4,
                ),
                " | val R2 = ",
                round(
                    validation_evaluation.metrics.r2;
                    digits = 4,
                ),
                " | grad norm = ",
                round(
                    current_gradient_norm;
                    digits = 4,
                ),
            )
        end
    end

    println()
    println("Training complete.")
    println(
        "Best epoch: ",
        best_epoch,
    )
    println(
        "Best validation loss: ",
        round(
            best_validation_loss;
            digits = 5,
        ),
    )

    return (
        parameters = best_ps,
        state = best_st,
        best_epoch = best_epoch,
        best_validation_loss =
            best_validation_loss,
    )
end


"""
    run_regression_experiment(...)

Train and evaluate one glucose reaction-property regression experiment.
"""
function run_regression_experiment(
    X,
    incidence_tail,
    incidence_head,
    targets,
    train_indices,
    validation_indices,
    test_indices;
    seed = RANDOM_SEED,
    epochs = NUMBER_OF_EPOCHS,
    num_layers::Int = 1,
    target_scaling = nothing,
)
    if isnothing(target_scaling)
        target_scaling =
            standardise_target(
                targets,
                train_indices,
            )
    end

    rng =
        MersenneTwister(seed)

    model =
        GlucoseHyperedgeRegressor(
            size(X, 2),
            HIDDEN_DIM;
            num_layers = num_layers,
        )

    ps, st =
        Lux.setup(
            rng,
            model,
        )

    println()
    println(
        "Message-passing layers: ",
        num_layers,
    )

    println(
        "Model parameter count: ",
        Lux.parameterlength(model),
    )

    println(
        "Training reactions: ",
        length(train_indices),
    )

    println(
        "Validation reactions: ",
        length(validation_indices),
    )

    println(
        "Test reactions: ",
        length(test_indices),
    )

    println()
    println(
        "Testing initial forward pass..."
    )

    initial_output,
    initial_state =
        regression_forward(
            model,
            (
                X,
                incidence_tail,
                incidence_head,
            ),
            ps,
            st,
        )

    println(
        "Hyperedge representation size: ",
        size(
            initial_output.hidden_hyperedges
        ),
    )

    println(
        "Prediction vector length: ",
        length(
            initial_output.predictions
        ),
    )

    expected_hyperedge_size =
        (
            size(incidence_tail, 2),
            HIDDEN_DIM,
        )

    size(
        initial_output.hidden_hyperedges
    ) == expected_hyperedge_size ||
        error(
            "Unexpected hyperedge representation size."
        )

    length(
        initial_output.predictions
    ) == size(incidence_tail, 2) ||
        error(
            "Unexpected number of reaction predictions."
        )

    all(
        isfinite,
        initial_output.predictions,
    ) || error(
        "Initial predictions contain non-finite values."
    )

    println(
        "Initial forward pass successful."
    )

    println()
    println(
        "Checking Enzyme gradient computation..."
    )

    initial_gradients =
        compute_regression_gradients(
            model,
            ps,
            initial_state,
            X,
            incidence_tail,
            incidence_head,
            target_scaling.values,
            train_indices,
            WEIGHT_DECAY,
        )

    parameter_tree_finite(
        initial_gradients
    ) || error(
        "Initial gradient check produced non-finite values."
    )

    println(
        "Initial gradient norm: ",
        round(
            gradient_norm(
                initial_gradients
            );
            digits = 6,
        ),
    )

    trained =
        train_regression_model(
            model,
            ps,
            initial_state,
            X,
            incidence_tail,
            incidence_head,
            target_scaling.values,
            targets,
            train_indices,
            validation_indices,
            target_scaling.mean,
            target_scaling.std;
            epochs = epochs,
        )

    final_test_evaluation =
        evaluate_regression_model(
            model,
            trained.parameters,
            trained.state,
            X,
            incidence_tail,
            incidence_head,
            target_scaling.values,
            targets,
            test_indices,
            target_scaling.mean,
            target_scaling.std,
        )

    return (
        model = model,
        parameters = trained.parameters,
        state = final_test_evaluation.state,
        best_epoch = trained.best_epoch,
        best_validation_loss =
            trained.best_validation_loss,
        test_metrics =
            final_test_evaluation.metrics,
        predictions =
            final_test_evaluation.predictions,
        target_scaling =
            target_scaling,
    )
end


"""
    save_regression_experiment(path, experiment; fit_indices, training_indices, validation_indices, test_indices)

Save a trained regression model and the metadata needed for later analysis.

The snapshot contains the model definition, best parameters, model state,
target-scaling statistics, split indices, best epoch, and test predictions.
Julia's standard `Serialization` module is used because these files are intended
for internal analysis with the same project environment.
"""
function save_regression_experiment(
    path,
    experiment;
    fit_indices,
    training_indices,
    validation_indices,
    test_indices,
)
    mkpath(
        dirname(path)
    )

    snapshot = (
        model = experiment.model,
        parameters = deepcopy(experiment.parameters),
        state = deepcopy(experiment.state),
        target_scaling = experiment.target_scaling,
        fit_indices = collect(fit_indices),
        training_indices = collect(training_indices),
        validation_indices = collect(validation_indices),
        test_indices = collect(test_indices),
        best_epoch = experiment.best_epoch,
        best_validation_loss =
            experiment.best_validation_loss,
        test_metrics = experiment.test_metrics,
        predictions = experiment.predictions,
    )

    open(
        path,
        "w",
    ) do io
        serialize(
            io,
            snapshot,
        )
    end

    return path
end


"""
    run_learning_curve(...)

Train the one-layer directed hypergraph regression model on increasing nested
subsets of one fixed training pool.

The train, validation, and test split is generated once and remains unchanged.
The training pool is shuffled once, and each larger learning-curve subset is a
prefix of the same ordering. Target standardisation is calculated once from the
complete training split and reused for every training-set size.

Each fitted model is evaluated post hoc on the same full training pool and on
the fixed validation and test sets. This keeps the evaluation population
consistent across training fractions. Mean signed error and MAE are reported in
the original reaction-energy scale. Each fitted model is also serialized for
later internal analysis.

# Returns

A vector of named tuples containing the training-set size, best epoch, and
train, validation, and test metrics for every learning-curve point.
"""
function run_learning_curve(
    X,
    incidence_tail,
    incidence_head,
    targets,
    split;
    percentages =
        LEARNING_CURVE_PERCENTAGES,
    seed = RANDOM_SEED,
)
    rng =
        MersenneTwister(seed)

    shuffled_training =
        copy(split.train)

    shuffle!(
        rng,
        shuffled_training,
    )

    fixed_target_scaling =
        standardise_target(
            targets,
            split.train,
        )

    results =
        NamedTuple[]

    model_directory =
        joinpath(
            @__DIR__,
            "saved_models",
        )

    println()
    println("Learning curve")
    println("Target property: DG")
    println("Energy unit: ", ENERGY_UNIT)
    println("Mean signed error = prediction - target")
    println(
        "Fixed training pool: ",
        length(split.train),
        " reactions",
    )
    println(
        "Fixed validation set: ",
        length(split.validation),
        " reactions",
    )
    println(
        "Fixed test set: ",
        length(split.test),
        " reactions",
    )
    println(
        "Post hoc training metrics are evaluated on the full fixed training pool."
    )

    for percentage in percentages
        percentage > 0 ||
            throw(
                ArgumentError(
                    "Learning-curve percentages must be positive."
                )
            )

        percentage <= 100 ||
            throw(
                ArgumentError(
                    "Learning-curve percentages cannot exceed 100."
                )
            )

        number_to_use =
            max(
                1,
                floor(
                    Int,
                    percentage /
                    100 *
                    length(shuffled_training),
                ),
            )

        selected_training =
            shuffled_training[
                1:number_to_use
            ]

        println()
        println(
            percentage,
            "% of available training reactions",
        )

        println(
            "Number of reactions used for fitting: ",
            number_to_use,
        )

        experiment =
            run_regression_experiment(
                X,
                incidence_tail,
                incidence_head,
                targets,
                selected_training,
                split.validation,
                split.test;
                seed = seed,
                num_layers = 1,
                target_scaling =
                    fixed_target_scaling,
            )

        training_evaluation =
            evaluate_regression_model(
                experiment.model,
                experiment.parameters,
                experiment.state,
                X,
                incidence_tail,
                incidence_head,
                fixed_target_scaling.values,
                targets,
                split.train,
                fixed_target_scaling.mean,
                fixed_target_scaling.std,
            )

        validation_evaluation =
            evaluate_regression_model(
                experiment.model,
                experiment.parameters,
                training_evaluation.state,
                X,
                incidence_tail,
                incidence_head,
                fixed_target_scaling.values,
                targets,
                split.validation,
                fixed_target_scaling.mean,
                fixed_target_scaling.std,
            )

        test_evaluation =
            evaluate_regression_model(
                experiment.model,
                experiment.parameters,
                validation_evaluation.state,
                X,
                incidence_tail,
                incidence_head,
                fixed_target_scaling.values,
                targets,
                split.test,
                fixed_target_scaling.mean,
                fixed_target_scaling.std,
            )

        model_path =
            joinpath(
                model_directory,
                "learning_curve_$(lpad(string(percentage), 3, '0'))pct.jls",
            )

        saved_path =
            save_regression_experiment(
                model_path,
                experiment;
                fit_indices =
                    selected_training,
                training_indices =
                    split.train,
                validation_indices =
                    split.validation,
                test_indices =
                    split.test,
            )

        push!(
            results,
            (
                percentage =
                    percentage,
                training_reactions =
                    number_to_use,
                best_epoch =
                    experiment.best_epoch,
                train_mean_signed_error =
                    training_evaluation.metrics.mean_signed_error,
                validation_mean_signed_error =
                    validation_evaluation.metrics.mean_signed_error,
                test_mean_signed_error =
                    test_evaluation.metrics.mean_signed_error,
                train_mae =
                    training_evaluation.metrics.mae,
                validation_mae =
                    validation_evaluation.metrics.mae,
                test_mae =
                    test_evaluation.metrics.mae,
                saved_model_path =
                    saved_path,
            ),
        )

        println(
            "Full-train ME = ",
            round(
                training_evaluation.metrics.mean_signed_error;
                digits = 4,
            ),
            " | Validation ME = ",
            round(
                validation_evaluation.metrics.mean_signed_error;
                digits = 4,
            ),
            " | Test ME = ",
            round(
                test_evaluation.metrics.mean_signed_error;
                digits = 4,
            ),
        )

        println(
            "Full-train MAE = ",
            round(
                training_evaluation.metrics.mae;
                digits = 4,
            ),
            " | Validation MAE = ",
            round(
                validation_evaluation.metrics.mae;
                digits = 4,
            ),
            " | Test MAE = ",
            round(
                test_evaluation.metrics.mae;
                digits = 4,
            ),
        )

        println(
            "Saved model: ",
            saved_path,
        )
    end

    return results
end


"""
    plot_learning_curve(learning_curve)

Create learning curves for mean signed error and mean absolute error.

Each graph shows training, validation, and test error against the percentage of
the available training set used. Error values are shown in the original
reaction-energy units. A horizontal zero line is included on the mean signed
error plot to distinguish average overprediction from underprediction.

# Arguments

- `learning_curve`: Results returned by `run_learning_curve`.

# Returns

A named tuple containing the mean signed error and MAE plots.
"""
function plot_learning_curve(learning_curve)
    percentages = [
        result.percentage
        for result in learning_curve
    ]

    train_mean_signed_error = [
        result.train_mean_signed_error
        for result in learning_curve
    ]

    validation_mean_signed_error = [
        result.validation_mean_signed_error
        for result in learning_curve
    ]

    test_mean_signed_error = [
        result.test_mean_signed_error
        for result in learning_curve
    ]

    train_mae = [
        result.train_mae
        for result in learning_curve
    ]

    validation_mae = [
        result.validation_mae
        for result in learning_curve
    ]

    test_mae = [
        result.test_mae
        for result in learning_curve
    ]

    mean_signed_error_plot =
        plot(
            percentages,
            train_mean_signed_error;
            marker = :circle,
            linewidth = 2,
            xlabel = "Training data used (% of training pool)",
            ylabel =
                "Mean signed error ($(ENERGY_UNIT))",
            title =
                "Glucose CRN Learning Curve - Mean Signed Error",
            label = "Full training pool",
            grid = true,
            xticks = percentages,
            xrotation = 45,
        )

    plot!(
        mean_signed_error_plot,
        percentages,
        validation_mean_signed_error;
        marker = :circle,
        linewidth = 2,
        label = "Validation",
    )

    plot!(
        mean_signed_error_plot,
        percentages,
        test_mean_signed_error;
        marker = :circle,
        linewidth = 2,
        label = "Test",
    )

    hline!(
        mean_signed_error_plot,
        [0.0];
        linestyle = :dash,
        linewidth = 1,
        label = "Zero error",
    )

    mae_plot =
        plot(
            percentages,
            train_mae;
            marker = :circle,
            linewidth = 2,
            xlabel = "Training data used (% of training pool)",
            ylabel = "MAE ($(ENERGY_UNIT))",
            title = "Glucose CRN Learning Curve - MAE",
            label = "Full training pool",
            grid = true,
            xticks = percentages,
            xrotation = 45,
        )

    plot!(
        mae_plot,
        percentages,
        validation_mae;
        marker = :circle,
        linewidth = 2,
        label = "Validation",
    )

    plot!(
        mae_plot,
        percentages,
        test_mae;
        marker = :circle,
        linewidth = 2,
        label = "Test",
    )

    mean_signed_error_output_path =
        joinpath(
            @__DIR__,
            "learning_curve_mean_signed_error.png",
        )

    mae_output_path =
        joinpath(
            @__DIR__,
            "learning_curve_mae.png",
        )

    savefig(
        mean_signed_error_plot,
        mean_signed_error_output_path,
    )

    savefig(
        mae_plot,
        mae_output_path,
    )

    println()
    println(
        "Mean signed error learning curve saved to: ",
        mean_signed_error_output_path,
    )
    println(
        "MAE learning curve saved to: ",
        mae_output_path,
    )

    return (
        mean_signed_error =
            mean_signed_error_plot,
        mae = mae_plot,
    )
end


"""
    main()

Run the glucose CRN message-passing depth comparison and learning-curve
evaluation.

The depth experiment compares one, two, and three directed message-passing
layers using exactly the same train, validation, and test split. The learning
curve then fits the one-layer model on nested fractions of the fixed training
pool while keeping preprocessing, evaluation populations, validation data, and
test data consistent. Trained model snapshots are saved under `saved_models`.

# Returns

A named tuple containing the depth-comparison results, learning-curve results,
generated plots, and the train/validation/test split.
"""
function main()
    Random.seed!(
        RANDOM_SEED
    )

    X =
        Float32.(
            vertex_features
        )

    incidence_tail =
        Matrix{Float32}(
            directed_crn.source_incidence
        )

    incidence_head =
        Matrix{Float32}(
            directed_crn.target_incidence
        )

    targets =
        Float32.(
            directed_crn.reaction_properties.DG
        )

    number_of_vertices =
        size(X, 1)

    number_of_features =
        size(X, 2)

    number_of_reactions =
        length(targets)

    size(
        incidence_tail
    ) == (
        number_of_vertices,
        number_of_reactions,
    ) || error(
        "Source incidence matrix has an unexpected size."
    )

    size(
        incidence_head
    ) == (
        number_of_vertices,
        number_of_reactions,
    ) || error(
        "Target incidence matrix has an unexpected size."
    )

    all(
        isfinite,
        X,
    ) || error(
        "Vertex features contain non-finite values."
    )

    all(
        isfinite,
        targets,
    ) || error(
        "DG target contains non-finite values."
    )

    split =
        random_regression_split(
            number_of_reactions;
            seed = RANDOM_SEED,
        )

    println()
    println(
        "Glucose CRN Message-Passing Depth Experiment"
    )
    println(
        "Target property: DG"
    )
    println(
        "Energy unit: ",
        ENERGY_UNIT,
    )
    println(
        "Molecular vertices: ",
        number_of_vertices,
    )
    println(
        "Vertex features: ",
        number_of_features,
    )
    println(
        "Reaction hyperedges: ",
        number_of_reactions,
    )
    println(
        "Training reactions: ",
        length(split.train),
    )
    println(
        "Validation reactions: ",
        length(split.validation),
    )
    println(
        "Test reactions: ",
        length(split.test),
    )
    println(
        "Hidden dimension: ",
        HIDDEN_DIM,
    )
    println(
        "Epochs per model: ",
        NUMBER_OF_EPOCHS,
    )
    println(
        "Same random seed and same split are used for all depths."
    )

    depth_results =
        NamedTuple[]

    for num_layers in (1, 2, 3)
        println()
        println(
            num_layers,
            num_layers == 1 ?
                " MESSAGE-PASSING LAYER" :
                " MESSAGE-PASSING LAYERS",
        )

        experiment =
            run_regression_experiment(
                X,
                incidence_tail,
                incidence_head,
                targets,
                split.train,
                split.validation,
                split.test;
                seed = RANDOM_SEED,
                epochs = NUMBER_OF_EPOCHS,
                num_layers = num_layers,
            )

        depth_model_path =
            joinpath(
                @__DIR__,
                "saved_models",
                "depth_$(num_layers)_layer$(num_layers == 1 ? "" : "s").jls",
            )

        saved_depth_model =
            save_regression_experiment(
                depth_model_path,
                experiment;
                fit_indices =
                    split.train,
                training_indices =
                    split.train,
                validation_indices =
                    split.validation,
                test_indices =
                    split.test,
            )

        push!(
            depth_results,
            (
                layers = num_layers,
                parameters =
                    Lux.parameterlength(
                        experiment.model
                    ),
                best_epoch =
                    experiment.best_epoch,
                validation_loss =
                    experiment.best_validation_loss,
                mean_signed_error =
                    experiment.test_metrics.mean_signed_error,
                mae =
                    experiment.test_metrics.mae,
                rmse =
                    experiment.test_metrics.rmse,
                r2 =
                    experiment.test_metrics.r2,
                saved_model_path =
                    saved_depth_model,
            ),
        )
    end

    println()
    println(
        "Message-passing depth comparison"
    )

    println(
        "Layers | Parameters | Best epoch | Test ME | Test MAE | Test RMSE | Test R2"
    )

    for result in depth_results
        println(
            lpad(
                string(result.layers),
                6,
            ),
            " | ",
            lpad(
                string(result.parameters),
                10,
            ),
            " | ",
            lpad(
                string(result.best_epoch),
                10,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.mean_signed_error;
                        digits = 4,
                    )
                ),
                7,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.mae;
                        digits = 4,
                    )
                ),
                8,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.rmse;
                        digits = 4,
                    )
                ),
                9,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.r2;
                        digits = 4,
                    )
                ),
                7,
            ),
        )
    end

    println()
    println("Saved depth models")
    for result in depth_results
        println(
            result.layers,
            result.layers == 1 ? " layer: " : " layers: ",
            result.saved_model_path,
        )
    end

    learning_curve_results =
        run_learning_curve(
            X,
            incidence_tail,
            incidence_head,
            targets,
            split,
        )

    learning_curve_plots =
        plot_learning_curve(
            learning_curve_results
        )

    return (
        depth_results = depth_results,
        learning_curve_results =
            learning_curve_results,
        learning_curve_plots =
            learning_curve_plots,
        split = split,
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end


