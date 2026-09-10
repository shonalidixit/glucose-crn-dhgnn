using Random
using Statistics
using LinearAlgebra
using Lux
using Optimisers
using Enzyme
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

const LEARNING_CURVE_PERCENTAGES =
    [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]


"""
    GlucoseHyperedgeRegressor

Directed hypergraph neural network for reaction-level regression.

The directed layer learns representations for molecular vertices and
reaction hyperedges. A linear regression head maps every learned
hyperedge representation to one scalar reaction property.
"""
struct GlucoseHyperedgeRegressor{L,IW,IB} <: Lux.AbstractLuxLayer
    directed_layer::L
    hidden_dim::Int
    init_weight::IW
    init_bias::IB
end


"""
    GlucoseHyperedgeRegressor(input_dim, hidden_dim)

Construct a directed hypergraph regression model.

The model does not require initial hyperedge features. Reaction
representations are learned from molecular vertex features together with
the source and target incidence matrices.
"""
function GlucoseHyperedgeRegressor(
    input_dim::Int,
    hidden_dim::Int;
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

    directed_layer =
        DirectedHypergraphLayer(
            input_dim,
            0,
            hidden_dim;
            activation = tanh,
            normalize = true,
        )

    return GlucoseHyperedgeRegressor(
        directed_layer,
        hidden_dim,
        init_weight,
        init_bias,
    )
end


"""
    Lux.initialparameters(rng, model)

Initialise the directed layer parameters and the scalar regression head.
"""
function Lux.initialparameters(
    rng::AbstractRNG,
    model::GlucoseHyperedgeRegressor,
)
    directed_parameters =
        Lux.initialparameters(
            rng,
            model.directed_layer,
        )

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
        directed_layer = directed_parameters,
        W_output = W_output,
        b_output = b_output,
    )
end


"""
    Lux.initialstates(rng, model)

Initialise state for the directed hypergraph layer.
"""
function Lux.initialstates(
    rng::AbstractRNG,
    model::GlucoseHyperedgeRegressor,
)
    directed_state =
        Lux.initialstates(
            rng,
            model.directed_layer,
        )

    return (
        directed_layer = directed_state,
    )
end


function Lux.parameterlength(
    model::GlucoseHyperedgeRegressor,
)
    directed_parameters =
        Lux.parameterlength(
            model.directed_layer,
        )

    output_parameters =
        model.hidden_dim + 1

    return directed_parameters +
           output_parameters
end


function Lux.statelength(
    model::GlucoseHyperedgeRegressor,
)
    return Lux.statelength(
        model.directed_layer,
    )
end


"""
    regression_forward(model, input, ps, st)

Run directed message passing and predict one scalar property for every
reaction hyperedge.
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

    directed_output,
    new_directed_state =
        model.directed_layer(
            (
                X,
                incidence_tail,
                incidence_head,
            ),
            ps.directed_layer,
            st.directed_layer,
        )

    hidden_hyperedges =
        directed_output.updated_hyperedges

    predictions =
        hidden_hyperedges *
        ps.W_output .+
        ps.b_output

    output = (
        predictions = vec(predictions),
        hidden_hyperedges = hidden_hyperedges,
    )

    new_state = (
        directed_layer = new_directed_state,
    )

    return output, new_state
end


"""
    random_regression_split(n; train_fraction, validation_fraction, seed)

Create reproducible train, validation and test splits over reaction
hyperedges.
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

    rng =
        MersenneTwister(seed)

    indices =
        collect(1:n)

    shuffle!(
        rng,
        indices,
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

    train_indices =
        indices[1:n_train]

    validation_start =
        n_train + 1

    validation_end =
        n_train + n_validation

    validation_indices =
        indices[
            validation_start:validation_end
        ]

    test_start =
        validation_end + 1

    test_indices =
        indices[
            test_start:end
        ]

    return (
        train = train_indices,
        validation = validation_indices,
        test = test_indices,
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

Calculate MAE, RMSE and R¬¨‚â§ for reaction-property predictions.
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
        mse = mse,
        mae = mae,
        rmse = rmse,
        r2 = r2,
    )
end


copy_parameter_tree(
    x::AbstractArray
) = copy(x)


copy_parameter_tree(
    x::NamedTuple
) =
    NamedTuple{keys(x)}(
        map(
            copy_parameter_tree,
            values(x),
        )
    )


copy_parameter_tree(
    x::Tuple
) =
    map(
        copy_parameter_tree,
        x,
    )


copy_parameter_tree(x) = x


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

Loss is calculated in standardised target space. MAE, RMSE and R¬¨‚â§ are
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
        copy_parameter_tree(
            current_ps
        )

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
                copy_parameter_tree(
                    current_ps
                )
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
                " | val R¬¨‚â§ = ",
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
        state = current_st,
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
)
    target_scaling =
        standardise_target(
            targets,
            train_indices,
        )

    rng =
        MersenneTwister(seed)

    model =
        GlucoseHyperedgeRegressor(
            size(X, 2),
            HIDDEN_DIM,
        )

    ps, st =
        Lux.setup(
            rng,
            model,
        )

    println()
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
    )
end


"""
    run_learning_curve(...)

Train the regression model using increasing percentages of the available
training reactions and evaluate each fitted model on its own training subset
and on the fixed validation set.

The test set is not used to construct the learning curve. It remains reserved
for the final model evaluation.
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

    results =
        NamedTuple[]

    println()
    println("Learning curve")
    println("--------------")

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
            "Number of training reactions: ",
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
            )

        target_scaling =
            standardise_target(
                targets,
                selected_training,
            )

        training_evaluation =
            evaluate_regression_model(
                experiment.model,
                experiment.parameters,
                experiment.state,
                X,
                incidence_tail,
                incidence_head,
                target_scaling.values,
                targets,
                selected_training,
                target_scaling.mean,
                target_scaling.std,
            )

        validation_evaluation =
            evaluate_regression_model(
                experiment.model,
                experiment.parameters,
                training_evaluation.state,
                X,
                incidence_tail,
                incidence_head,
                target_scaling.values,
                targets,
                split.validation,
                target_scaling.mean,
                target_scaling.std,
            )

        push!(
            results,
            (
                percentage =
                    percentage,
                training_reactions =
                    number_to_use,
                train_mse =
                    training_evaluation.metrics.mse,
                validation_mse =
                    validation_evaluation.metrics.mse,
                train_mae =
                    training_evaluation.metrics.mae,
                validation_mae =
                    validation_evaluation.metrics.mae,
            ),
        )

        println(
            "Train MSE = ",
            round(
                training_evaluation.metrics.mse;
                digits = 4,
            ),
            " | Validation MSE = ",
            round(
                validation_evaluation.metrics.mse;
                digits = 4,
            ),
            " | Train MAE = ",
            round(
                training_evaluation.metrics.mae;
                digits = 4,
            ),
            " | Validation MAE = ",
            round(
                validation_evaluation.metrics.mae;
                digits = 4,
            ),
        )
    end

    return results
end


"""
    plot_learning_curve(learning_curve)

Create conventional learning curves for MSE and MAE.

Each graph shows training and validation error against the percentage of the
available training set used. Lower values indicate better predictions.
"""
function plot_learning_curve(learning_curve)
    percentages = [
        result.percentage
        for result in learning_curve
    ]

    train_mse = [
        result.train_mse
        for result in learning_curve
    ]

    validation_mse = [
        result.validation_mse
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

    mse_plot =
        plot(
            percentages,
            train_mse;
            marker = :circle,
            linewidth = 2,
            xlabel = "Training data (%)",
            ylabel = "MSE",
            title = "Glucose CRN Learning Curve - MSE",
            label = "Training",
            grid = true,
            xticks = percentages,
            xrotation = 45,
        )

    plot!(
        mse_plot,
        percentages,
        validation_mse;
        marker = :circle,
        linewidth = 2,
        label = "Validation",
    )

    mae_plot =
        plot(
            percentages,
            train_mae;
            marker = :circle,
            linewidth = 2,
            xlabel = "Training data (%)",
            ylabel = "MAE",
            title = "Glucose CRN Learning Curve - MAE",
            label = "Training",
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

    mse_output_path =
        joinpath(
            @__DIR__,
            "learning_curve_mse.png",
        )

    mae_output_path =
        joinpath(
            @__DIR__,
            "learning_curve_mae.png",
        )

    savefig(
        mse_plot,
        mse_output_path,
    )

    savefig(
        mae_plot,
        mae_output_path,
    )

    println()
    println(
        "MSE learning curve saved to: ",
        mse_output_path,
    )
    println(
        "MAE learning curve saved to: ",
        mae_output_path,
    )

    return (
        mse = mse_plot,
        mae = mae_plot,
    )
end


"""
    main()

Run the glucose directed-hypergraph reaction regression proof of concept.
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
        "Glucose CRN Hyperedge Regression"
    )
    println(
        "--------------------------------"
    )
    println(
        "Target property: DG"
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
        "Source incidence matrix: ",
        size(incidence_tail),
    )
    println(
        "Target incidence matrix: ",
        size(incidence_head),
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

    println()
    println(
        "Full training-set experiment"
    )
    println(
        "----------------------------"
    )

    experiment =
        run_regression_experiment(
            X,
            incidence_tail,
            incidence_head,
            targets,
            split.train,
            split.validation,
            split.test,
        )

    println()
    println(
        "Final test performance"
    )
    println(
        "----------------------"
    )
    println(
        "Best epoch: ",
        experiment.best_epoch,
    )
    println(
        "MSE: ",
        round(
            experiment.test_metrics.mse;
            digits = 4,
        ),
    )
    println(
        "MAE: ",
        round(
            experiment.test_metrics.mae;
            digits = 4,
        ),
    )
    println(
        "RMSE: ",
        round(
            experiment.test_metrics.rmse;
            digits = 4,
        ),
    )
    println(
        "R¬¨‚â§: ",
        round(
            experiment.test_metrics.r2;
            digits = 4,
        ),
    )

    learning_curve =
        run_learning_curve(
            X,
            incidence_tail,
            incidence_head,
            targets,
            split,
        )

    learning_curve_plot =
        plot_learning_curve(
            learning_curve
        )

    println()
    println(
        "Learning curve summary"
    )
    println(
        "----------------------"
    )

    for result in learning_curve
        println(
            lpad(
                string(
                    result.percentage
                ) * "%",
                4,
            ),
            " | n = ",
            lpad(
                string(
                    result.training_reactions
                ),
                4,
            ),
            " | train MSE = ",
            round(
                result.train_mse;
                digits = 4,
            ),
            " | val MSE = ",
            round(
                result.validation_mse;
                digits = 4,
            ),
            " | train MAE = ",
            round(
                result.train_mae;
                digits = 4,
            ),
            " | val MAE = ",
            round(
                result.validation_mae;
                digits = 4,
            ),
        )
    end

    return (
        experiment = experiment,
        learning_curve = learning_curve,
        learning_curve_plot = learning_curve_plot,
    )
end


results = main()

