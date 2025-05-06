import os
from pathlib import Path


def add_replicator_args(parser):
    parser.add_argument(
        "--scenario",
        required=False,
        default=None,
        help="scenario name or path. (default: None)",
    )

    parser.add_argument(
        "-l",
        "--list-scenes",
        action="store_true",
        required=False,
        default=False,
        help="List available scenario presets.",
    )

    parser.add_argument(
        "--num-frames",
        required=False,
        # While default is truly 100, we set to 0 here in order to support legacy arg syntax
        # that is, default is applied later
        default=0,
        help="Frame numbers (default: 100)",
    )
    parser.add_argument(
        "-w",
        "--writer",
        required=False,
        default="BasicDSWriterFull",
        help="Writer (default: BasicDSWriterFull)",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=False,
        default=None,
        help="Set output path.",
    )
    parser.add_argument(
        "--rand-config",
        required=False,
        default=None,
        help="Name of randomization TOML file. (default: None)",
    )
    parser.add_argument(
        "--rand-config-dir",
        required=False,
        default=None,
        help="Path to the folder that holds randomization TOML file. \
            (default: Read configs inside the DS2 container \
            /drivesim-ov/_build/linux-x86_64/release/exts/omni.drivesim.replicator.domainrand\
            /python/omni/drivesim/replicator/domainrand/config)",
    )
    parser.add_argument(
        "--rand-asset-dir",
        required=False,
        default=None,
        help="Path to the folder that holds asset files to be used with randomization TOML file. \
            (default: Read configs inside the DS2 container \
            /drivesim-ov/_build/linux-x86_64/release/exts/omni.drivesim.replicator.domainrand\
            /python/omni/drivesim/replicator/domainrand/config/assets)",
    )
    parser.add_argument(
        "--show-ui",
        action="store_true",
        required=False,
        default=False,
        help="Show ui. (default: %(default)s)",
    )
    parser.add_argument(
        "--run-count",
        "-r",
        required=False,
        default=1,
        help="Number of frame generation sessions to schedule.",
        type=int,
    )
    parser.add_argument(
        "--ds-config",
        required=False,
        default="release",
        help="Datastudio build configuration. [debug|release](default: %(default)s)",
    )
    parser.add_argument(
        "--kit-config",
        required=False,
        default="release",
        help="Kit build configuration. [debug|release](default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--render-level",
        required=False,
        default="ultralow",
        help="Rendering level preset. [ultralow, low, medium, high, ultra](default: %(default)s)",
    )
    parser.add_argument(
        "--allow-root",
        action="store_true",
        required=False,
        default=False,
        help="Allow root. (default: %(default)s)",
    )
    parser.add_argument(
        "--texture-streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable texture streaming (for VRAM-constrained environments) (default: %(default)s)",
    )
    parser.add_argument(
        "--no-livestream",
        required=False,
        default=False,
        help="Disable livestream (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--smart-display-env",
        required=False,
        action="store_true",
        default=False,
        help="Smartly set DISPLAY variable before launching",
    )
    parser.add_argument(
        "--nucleus-path",
        required=False,
        default=None,
        help="Set nucleus path. (default: omniverse://drivesim2-rel.ov.nvidia.com)",
    )
    parser.add_argument(
        "-p",
        "--print-command",
        required=False,
        action="store_true",
        default=False,
        help="Print kit command before running",
    )

    parser.add_argument(
        "--print-command-only",
        required=False,
        action="store_true",
        default=False,
        help="Print kit command but don't run",
    )
    parser.add_argument(
        "--custom-env-setting",
        required=False,
        nargs="?",
        const="omni_env_settings.json",
        help="Introduce a config json file to load environment variables",
    )
    parser.add_argument(
        "--monitor-usd",
        required=False,
        action="store_true",
        default=False,
        help="Monitors USD for modifications during runtime",
    )
    parser.add_argument(
        "--seed-value",
        required=False,
        default=None,
        help="Set seed value for repeatable randomization",
    )
    parser.add_argument(
        "--manual-play",
        required=False,
        action="store_true",
        default=False,
        help="Disable automatic start of autotest script and launch Datastudio GUI for debugging",
    )
    parser.add_argument(
        "--disable-autotest",
        required=False,
        action="store_true",
        default=False,
        help="Disable autotest script and launch Datastudio GUI for debugging",
    )
    parser.add_argument(
        "--log-file",
        required=False,
        default=None,
        help="Path to alternative logging directory.",
        type=Path,
    )

    parser.add_argument(
        "--profile",
        required=False,
        action="store_true",
        default=False,
        help="Enable Tracy profiling",
    )

    parser.add_argument(
        "--usd-teleportation",
        required=False,
        action="store_true",
        default=False,
        help="Switch all prim teleportation to USD",
    )
    parser.add_argument(
        "--export-usd-log-path",
        required=False,
        default=None,
        help="Export USD log to following relative or abs path (token resolution not supported)",
        type=os.path.abspath,
    )

    return parser
