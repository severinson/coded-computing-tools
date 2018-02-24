# simulate rateless code performance. stores the result as a .csv file.
# required Julia packages: ArgParse, RaptorCodes

using RaptorCodes, ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--code"
        help = "code type. LT or R10."
        arg_type = String
        required = true
        "--num_inputs"
        help = "number of input/source symbols"
        arg_type = Int
        required = true
        "--mode"
        help = "location of the robust Soliton distribution spike"
        arg_type = Int
        required = true
        "--delta"
        help = "failure probability parameter for the robust Soliton distribution"
        arg_type = Float64
        required = true
        "--overhead"
        help = "relative reception overhead"
        arg_type = Float64
        required = true
        "--samples"
        help = "number of performance samples"
        arg_type = Int
        default = 100
        "--write"
        help = "write results to this file"
        arg_type = String
        default = ""
    end
    return parse_args(s)
end

function main()
    println(ARGS)
    args = parse_commandline()
    if args["code"] == "LT"
        if !("mode" in keys(args))
            error("mode must be provided for LT codes")
        end
        if !("delta" in keys(args))
            error("mode must be provided for LT codes")
        end
        p = LTParameters(
            args["num_inputs"],
            Soliton(args["num_inputs"], args["mode"], args["delta"]),
        )
    elseif args["code"] == "R10"
        p = R10Parameters(args["num_inputs"])
    else
        error("--code must be LT or R10")
    end
    df = RaptorCodes.simulate(
        p,
        args["overhead"],
        args["samples"],
    )
    if length(args["write"]) > 0
        mkpath(dirname(args["write"]))
        CSV.write(args["write"], df)
    end
    return
end

main()
