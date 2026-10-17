"""Fetch the native SDK from checksum-pinned NVIDIA wheels; no Python runtime."""

def _libcudf_impl(ctx):
    includes = []
    for source in json.decode(ctx.read(ctx.attr.sources)):
        name = source["name"]
        package = "packages/" + name
        ctx.download_and_extract(
            url = source["url"],
            sha256 = source["sha256"],
            type = "zip",
            output = package,
        )
        if source.get("format") == "conda":
            for archive in ctx.path(package).readdir():
                if archive.basename.startswith("pkg-"):
                    ctx.extract(archive, output = package, type = "tar.zst")
        root = ctx.path(package + "/" + source["root"])
        if root.get_child("include").exists:
            include = "include/" + name
            ctx.symlink(root.get_child("include"), include)
            if root.get_child("include/rapids").exists:
                includes.append(include + "/rapids")
            includes.append(include)
        directories = [root.get_child("lib"), root.get_child("lib64")]
        directories.extend([p for p in ctx.path(package).readdir() if p.basename.endswith(".libs")])
        for directory in directories:
            if directory.exists:
                for library in directory.readdir():
                    if (
                        ".so" in library.basename and
                        ".alt." not in library.basename and
                        library.basename in source.get("libraries", [library.basename])
                    ):
                        ctx.symlink(library, "lib/" + library.basename)
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "libcudf",
    hdrs = glob(["include/**"]),
    srcs = glob(["lib/*.so*"]),
    includes = %s,
    visibility = ["//visibility:public"],
)
""" % repr(includes))

libcudf_repository = repository_rule(
    implementation = _libcudf_impl,
    attrs = {"sources": attr.label(allow_single_file = [".json"], mandatory = True)},
)
