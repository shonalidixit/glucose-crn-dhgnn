"""
    learning_curve_signed_error.jl

Learning-curve evaluation for the glucose chemical reaction network directed
hypergraph neural network.

The experiment measures model performance as the amount of available training
data increases. A single reproducible train/validation/test split is used
throughout the experiment. Training subsets are nested so that every larger
subset contains the reactions used by the preceding smaller subsets.

The model is trained using mean squared error in standardised target space.
Performance is reported using mean signed error and mean absolute error in the
original ΔG scale.

Mean signed error is defined as

    prediction - target

so positive values indicate overprediction and negative values indicate
underprediction.

Target standardisation is calculated once using the complete training split and
is reused for every learning-curve point. The validation and test sets remain
fixed for all experiments.
"""

using Random
using Statistics
using Lux
using Plots

include("message_passing_depth_experiment.jl")


"""
Energy unit used for the reaction free-energy target.
"""
const ENERGY_UNIT = "kcal/mol"


"""
Percentages of the available training split used to construct the learning
curve.
"""
const SIGNED_ERROR_LEARNING_CURVE_PERCENTAGES =
    [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]


"""
    mean_signed_error(predictions, targets, indices)

Calculate the mean signed prediction error for a selected group of reactions.

The signed error for reaction `i` is

    predictions[i] - targets[i]

A positive result means that predictions are, on average, greater than the
observed values. A negative result means that predictions are, on average,
smaller than the observed values.

# Arguments

- `predictions`: Model predictions in the original target scale.
- `targets`: Observed reaction-property values.
- `indices`: Reaction indices included in the calculation.

# Returns

The mean signed error over the selected reactions.
"""
function mean_signed_error(
    predictions,
    targets,
    indices,
)
    isempty(indices) &&
        throw(
            ArgumentError(
                "Cannot calculate mean signed error for an empty index set."
            )
        )

    residuals =
        predictions[indices] .-
        targets[indices]

    return mean(residuals)
end


"""
    evaluate_learning_curve_split(
        model,
        ps,
        st,
        X,
        incidence_tail,
        incidence_head,
        targets,
        indices,
        target_mean,
        target_std,
    )

Evaluate a trained directed hypergraph regression model on one reaction split.

The model produces predictions in standardised target space. These predictions
are converted back to the original ΔG scale before mean signed error and mean
absolute error are calculated.

# Arguments

- `model`: Directed hypergraph regression model.
- `ps`: Trained Lux model parameters.
- `st`: Lux model state.
- `X`: Molecular vertex-feature matrix.
- `incidence_tail`: Source-side directed incidence matrix.
- `incidence_head`: Target-side directed incidence matrix.
- `targets`: Reaction-property targets in the original scale.
- `indices`: Reaction indices to evaluate.
- `target_mean`: Mean used for target standardisation.
- `target_std`: Standard deviation used for target standardisation.

# Returns

A named tuple containing:

- `mean_signed_error`: Mean prediction minus target error.
- `mae`: Mean absolute error.
- `predictions`: Predictions in the original target scale.
- `state`: Updated Lux model state.
"""
function evaluate_learning_curve_split(
    model,
    ps,
    st,
    X,
    incidence_tail,
    incidence_head,
    targets,
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

    predictions =
        output.predictions .*
        target_std .+
        target_mean

    signed_error =
        mean_signed_error(
            predictions,
            targets,
            indices,
        )

    absolute_error =
        mean(
            abs.(
                predictions[indices] .-
                targets[indices]
            )
        )

    return (
        mean_signed_error = signed_error,
        mae = absolute_error,
        predictions = predictions,
        state = new_state,
    )
end


"""
    run_signed_error_learning_curve(
        X,
        incidence_tail,
        incidence_head,
        targets,
        split;
        percentages=SIGNED_ERROR_LEARNING_CURVE_PERCENTAGES,
        seed=RANDOM_SEED,
    )

Construct a learning curve for the one-layer directed hypergraph regression
model.

One ordering of the training split is generated using the supplied random seed.
Each learning-curve training set is a prefix of this ordering. Consequently,
training subsets are nested and differ only in the number of reactions used.

Target standardisation is calculated from the complete training split before
the learning-curve experiments begin. The same scaling is then used for every
training subset.

The validation and test sets are held fixed throughout the experiment. A new
model is initialised with the same seed for every training percentage.

Model optimisation continues to use mean squared error in standardised target
space. Mean signed error and mean absolute error are calculated after
predictions are transformed back to the original ΔG scale.

# Arguments

- `X`: Molecular vertex-feature matrix.
- `incidence_tail`: Directed source incidence matrix.
- `incidence_head`: Directed target incidence matrix.
- `targets`: Reaction ΔG values.
- `split`: Named tuple containing `train`, `validation`, and `test` indices.
- `percentages`: Percentages of the training split to evaluate.
- `seed`: Random seed used for training-set ordering and model initialisation.

# Returns

A vector of named tuples containing the training-set size, best epoch, and
train/validation/test error measurements for every learning-curve point.
"""
function run_signed_error_learning_curve(
    X,
    incidence_tail,
    incidence_head,
    targets,
    split;
    percentages =
        SIGNED_ERROR_LEARNING_CURVE_PERCENTAGES,
    seed = RANDOM_SEED,
)
    rng =
        MersenneTwister(seed)

    training_order =
        copy(split.train)

    shuffle!(
        rng,
        training_order,
    )

    target_scaling =
        standardise_target(
            targets,
            split.train,
        )

    results =
        NamedTuple[]

    println()
    println("Glucose CRN learning curve")
    println("Target property: DG")
    println("Energy unit: ", ENERGY_UNIT)
    println("Signed error definition: prediction - target")
    println(
        "Available training reactions: ",
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

    for percentage in percentages
        0 < percentage <= 100 ||
            throw(
                ArgumentError(
                    "Learning-curve percentages must be between 1 and 100."
                )
            )

        number_to_use =
            max(
                1,
                floor(
                    Int,
                    percentage /
                    100 *
                    length(training_order),
                ),
            )

        selected_training =
            training_order[
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

        model_rng =
            MersenneTwister(seed)

        model =
            GlucoseHyperedgeRegressor(
                size(X, 2),
                HIDDEN_DIM;
                num_layers = 1,
            )

        ps,
        st =
            Lux.setup(
                model_rng,
                model,
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

        all(
            isfinite,
            initial_output.predictions,
        ) || error(
            "Initial predictions contain non-finite values."
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
                selected_training,
                WEIGHT_DECAY,
            )

        parameter_tree_finite(
            initial_gradients
        ) || error(
            "Initial gradient check produced non-finite values."
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
                selected_training,
                split.validation,
                target_scaling.mean,
                target_scaling.std;
                epochs = NUMBER_OF_EPOCHS,
            )

        training_evaluation =
            evaluate_learning_curve_split(
                model,
                trained.parameters,
                trained.state,
                X,
                incidence_tail,
                incidence_head,
                targets,
                selected_training,
                target_scaling.mean,
                target_scaling.std,
            )

        validation_evaluation =
            evaluate_learning_curve_split(
                model,
                trained.parameters,
                training_evaluation.state,
                X,
                incidence_tail,
                incidence_head,
                targets,
                split.validation,
                target_scaling.mean,
                target_scaling.std,
            )

        test_evaluation =
            evaluate_learning_curve_split(
                model,
                trained.parameters,
                validation_evaluation.state,
                X,
                incidence_tail,
                incidence_head,
                targets,
                split.test,
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
                train_mean_signed_error =
                    training_evaluation.mean_signed_error,
                validation_mean_signed_error =
                    validation_evaluation.mean_signed_error,
                test_mean_signed_error =
                    test_evaluation.mean_signed_error,
                train_mae =
                    training_evaluation.mae,
                validation_mae =
                    validation_evaluation.mae,
                test_mae =
                    test_evaluation.mae,
                best_epoch =
                    trained.best_epoch,
            ),
        )

        println(
            "Train mean signed error = ",
            round(
                training_evaluation.mean_signed_error;
                digits = 4,
            ),
            " | Validation mean signed error = ",
            round(
                validation_evaluation.mean_signed_error;
                digits = 4,
            ),
            " | Test mean signed error = ",
            round(
                test_evaluation.mean_signed_error;
                digits = 4,
            ),
        )

        println(
            "Train MAE = ",
            round(
                training_evaluation.mae;
                digits = 4,
            ),
            " | Validation MAE = ",
            round(
                validation_evaluation.mae;
                digits = 4,
            ),
            " | Test MAE = ",
            round(
                test_evaluation.mae;
                digits = 4,
            ),
        )

        println(
            "Best epoch: ",
            trained.best_epoch,
        )
    end

    return results
end


"""
    plot_signed_error_learning_curve(results)

Create learning-curve plots for mean signed error and mean absolute error.

Both figures contain training, validation, and test measurements for each
training-set percentage.

Mean signed error includes a horizontal line at zero. Values above zero indicate
average overprediction, while values below zero indicate average
underprediction.

The figures are saved in the same directory as this source file.

# Arguments

- `results`: Output returned by [`run_signed_error_learning_curve`](@ref).

# Returns

A named tuple containing the mean signed error plot and MAE plot.
"""
function plot_signed_error_learning_curve(
    results,
)
    percentages =
        [
            result.percentage
            for result in results
        ]

    train_signed_error =
        [
            result.train_mean_signed_error
            for result in results
        ]

    validation_signed_error =
        [
            result.validation_mean_signed_error
            for result in results
        ]

    test_signed_error =
        [
            result.test_mean_signed_error
            for result in results
        ]

    train_mae =
        [
            result.train_mae
            for result in results
        ]

    validation_mae =
        [
            result.validation_mae
            for result in results
        ]

    test_mae =
        [
            result.test_mae
            for result in results
        ]

    signed_error_plot =
        plot(
            percentages,
            train_signed_error;
            marker = :circle,
            linewidth = 2,
            xlabel = "Training data (%)",
            ylabel = "Mean signed error ($(ENERGY_UNIT))",
            title = "Glucose CRN Learning Curve - Mean Signed Error",
            label = "Training",
            grid = true,
            xticks = percentages,
            xrotation = 45,
        )

    plot!(
        signed_error_plot,
        percentages,
        validation_signed_error;
        marker = :circle,
        linewidth = 2,
        label = "Validation",
    )

    plot!(
        signed_error_plot,
        percentages,
        test_signed_error;
        marker = :circle,
        linewidth = 2,
        label = "Test",
    )

    hline!(
        signed_error_plot,
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
            xlabel = "Training data (%)",
            ylabel = "MAE ($(ENERGY_UNIT))",
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

    plot!(
        mae_plot,
        percentages,
        test_mae;
        marker = :circle,
        linewidth = 2,
        label = "Test",
    )

    signed_error_output_path =
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
        signed_error_plot,
        signed_error_output_path,
    )

    savefig(
        mae_plot,
        mae_output_path,
    )

    println()
    println(
        "Mean signed error learning curve saved to: ",
        signed_error_output_path,
    )
    println(
        "MAE learning curve saved to: ",
        mae_output_path,
    )

    return (
        mean_signed_error = signed_error_plot,
        mae = mae_plot,
    )
end


"""
    print_learning_curve_results(results)

Print the learning-curve measurements in tabular form.

For every training percentage, the table reports the number of training
reactions together with train, validation, and test mean signed error and mean
absolute error.

# Arguments

- `results`: Output returned by [`run_signed_error_learning_curve`](@ref).

# Returns

`nothing`.
"""
function print_learning_curve_results(
    results,
)
    println()
    println("Learning Curve Results")
    println()

    println(
        "Data % | N train | Train ME | Val ME | Test ME | Train MAE | Val MAE | Test MAE"
    )

    for result in results
        println(
            lpad(
                string(result.percentage),
                6,
            ),
            " | ",
            lpad(
                string(result.training_reactions),
                7,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.train_mean_signed_error;
                        digits = 4,
                    )
                ),
                8,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.validation_mean_signed_error;
                        digits = 4,
                    )
                ),
                6,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.test_mean_signed_error;
                        digits = 4,
                    )
                ),
                7,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.train_mae;
                        digits = 4,
                    )
                ),
                9,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.validation_mae;
                        digits = 4,
                    )
                ),
                7,
            ),
            " | ",
            lpad(
                string(
                    round(
                        result.test_mae;
                        digits = 4,
                    )
                ),
                8,
            ),
        )
    end

    println()
    println(
        "Mean signed error = prediction - target"
    )
    println(
        "Energy unit: ",
        ENERGY_UNIT,
    )

    return nothing
end


"""
    main_signed_error_learning_curve()

Run the complete glucose CRN learning-curve experiment.

The molecular fingerprint matrix and directed incidence matrices are loaded from
the glucose CRN preprocessing code. Reaction ΔG is used as the regression
target.

A reproducible train/validation/test split is created before training. The
learning curve is then evaluated using increasing nested subsets of the training
split. Results are printed and the mean signed error and MAE figures are saved.

# Returns

A named tuple containing:

- `results`: Learning-curve measurements.
- `plots`: Generated Plots.jl figures.
- `split`: Train, validation, and test reaction indices.
"""
function main_signed_error_learning_curve()
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
        "Glucose CRN Signed-Error Learning Curve"
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

    learning_curve =
        run_signed_error_learning_curve(
            X,
            incidence_tail,
            incidence_head,
            targets,
            split,
        )

    print_learning_curve_results(
        learning_curve
    )

    plots =
        plot_signed_error_learning_curve(
            learning_curve
        )

    return (
        results = learning_curve,
        plots = plots,
        split = split,
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    signed_error_learning_curve_results =
        main_signed_error_learning_curve()
end