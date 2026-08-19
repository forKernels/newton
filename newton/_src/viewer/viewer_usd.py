# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from typing import Any

import numpy as np
import warp as wp

import newton

from ..core.types import override

if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
    _usd_root = os.environ.get("OPENUSD_ROOT") or os.environ.get("USD_ROOT")
    if _usd_root and os.path.isdir(_usd_root):
        for _subdir in ("bin", "lib"):
            _dll_dir = os.path.join(_usd_root, _subdir)
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
except ImportError as e:
    Gf = Sdf = Usd = UsdGeom = UsdShade = Vt = None
    _PXR_IMPORT_ERROR = e
else:
    _PXR_IMPORT_ERROR = None

from .viewer import _DEFAULT_LAYER_ID, ViewerBase


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


def _usd_add_xform(prim):
    prim = UsdGeom.Xform(prim)
    prim.ClearXformOpOrder()

    prim.AddTranslateOp()
    prim.AddOrientOp()
    prim.AddScaleOp()


def _usd_set_xform(
    xform,
    pos: tuple | None = None,
    rot: tuple | None = None,
    scale: tuple | None = None,
    time: float = 0.0,
):
    xform = UsdGeom.Xform(xform)

    xform_ops = xform.GetOrderedXformOps()

    if pos is not None:
        xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
    if rot is not None:
        xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    if scale is not None:
        xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)


class ViewerUSD(ViewerBase):
    """
    USD viewer backend for Newton physics simulations.

    This backend creates a USD stage and manages mesh prototypes and instanced rendering
    using PointInstancers. It supports time-sampled transforms for efficient playback
    and visualization of simulation data.
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 60,
        up_axis: str = "Z",
        num_frames: int | None = 100,
        scaling: float = 1.0,
        points_as_spheres: bool = True,
    ):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path: Path to the output USD file.
            fps: Frames per second for time sampling. Default is 60.
            up_axis: USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames: Maximum number of frames to record. Default is 100. If None, recording is unlimited.
            scaling: Uniform scaling applied to the scene root. Default is 1.0.
            points_as_spheres: When True, :meth:`log_points` renders points as a
                :class:`~pxr.UsdGeom.PointInstancer` of :class:`~pxr.UsdGeom.Sphere`
                prototypes scaled by ``radii``, instead of flat
                :class:`~pxr.UsdGeom.Points` splats. Default is True.

        Raises:
            ImportError: If the usd-core package is not installed.
        """
        if Usd is None:
            raise ImportError(
                "usd-core/pxr is unavailable for ViewerUSD. On Windows with a local OpenUSD install, set "
                "OPENUSD_ROOT (or USD_ROOT) so <root>\\bin can be added to the DLL search path."
            ) from _PXR_IMPORT_ERROR

        super().__init__()

        self.output_path = os.path.abspath(output_path)
        self.fps = fps
        self.up_axis = up_axis
        self.scaling = scaling
        self.num_frames = num_frames
        self.points_as_spheres = points_as_spheres

        # Create USD stage. If this output path is already registered in the
        # current process, reuse and clear the existing layer instead of
        # calling CreateNew() again (which raises for duplicate identifiers).
        existing_layer = Sdf.Layer.Find(self.output_path)
        if existing_layer is not None:
            existing_layer.Clear()
            self.stage = Usd.Stage.Open(existing_layer)
        else:
            self.stage = Usd.Stage.CreateNew(self.output_path)
        self.stage.SetTimeCodesPerSecond(fps)  # number of timeCodes per second for data storage
        self.stage.SetFramesPerSecond(fps)  # display frame rate (timeline FPS in DCC tools)
        self.stage.SetStartTimeCode(0)

        axis_token = {
            "X": UsdGeom.Tokens.x,
            "Y": UsdGeom.Tokens.y,
            "Z": UsdGeom.Tokens.z,
        }.get(self.up_axis.strip().upper())

        UsdGeom.SetStageUpAxis(self.stage, axis_token)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        self.root = UsdGeom.Xform.Define(self.stage, "/root")

        # apply root scaling
        self.root.ClearXformOpOrder()
        s = self.root.AddScaleOp()
        s.Set(Gf.Vec3d(float(scaling), float(scaling), float(scaling)), 0.0)

        self.stage.SetDefaultPrim(self.root.GetPrim())

        # textures/ directory path, shared across layers (created lazily)
        self._texture_dir = None

        # Track current frame
        self._frame_index = 0
        self._frame_count = 0

        self.set_model(None)

    @override
    def _init_extra_layer_state(self, layer) -> None:
        super()._init_extra_layer_state(layer)
        layer._meshes = {}  # mesh_name -> prototype path
        layer._instance_groups = {}  # instance_name -> group prim for individually referenced meshes
        layer._instancers = {}  # instancer_name -> UsdGeom.PointInstancer
        layer._points = {}  # point_name -> UsdGeom.Points
        layer._texture_materials: dict[str, Any] = {}  # mesh_name -> UsdShade.Material
        layer._mesh_data = {}  # mesh_name -> topology dict for hidden prototypes (not written to USD)
        layer._materials = {}  # material_key -> USD material path
        layer._mesh_materials = {}  # mesh prototype name -> (color, roughness, metallic, texture)

    def _reset_stage(self):
        self.stage.GetRootLayer().Clear()
        self.stage.SetTimeCodesPerSecond(self.fps)
        self.stage.SetFramesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        axis_token = {
            "X": UsdGeom.Tokens.x,
            "Y": UsdGeom.Tokens.y,
            "Z": UsdGeom.Tokens.z,
        }.get(self.up_axis.strip().upper())
        UsdGeom.SetStageUpAxis(self.stage, axis_token)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.root = UsdGeom.Xform.Define(self.stage, "/root")
        self.root.ClearXformOpOrder()
        s = self.root.AddScaleOp()
        s.Set(Gf.Vec3d(float(self.scaling), float(self.scaling), float(self.scaling)), 0.0)
        self.stage.SetDefaultPrim(self.root.GetPrim())
        self._frame_index = 0
        self._frame_count = 0

    @staticmethod
    def _save_texture_atomic(tex_array: np.ndarray, tex_path: str) -> None:
        """Write a texture image atomically so readers never see a partial PNG."""
        from PIL import Image

        tex_dir = os.path.dirname(tex_path)
        base_name = os.path.basename(tex_path)
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=f".{base_name}.", suffix=".tmp", dir=tex_dir)
            os.close(fd)
            Image.fromarray(tex_array).save(tmp_path, format="PNG")
            os.replace(tmp_path, tex_path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _remove_active_layer_prims(self):
        names = set(self._meshes) | set(self._instance_groups) | set(self._instancers) | set(self._points)
        for name in sorted(names, key=lambda item: self._get_path(item).count("/"), reverse=True):
            if self._is_layer_owned_path(name):
                self.stage.RemovePrim(self._get_path(name))

        for mesh_name in list(self._texture_materials):
            if self._is_layer_owned_path(mesh_name):
                self.stage.RemovePrim(self._texture_material_path(mesh_name))

    def _has_user_layers(self) -> bool:
        return any(layer_id != _DEFAULT_LAYER_ID for layer_id in self._layers)

    def clear_model(self):
        if hasattr(self, "stage") and self.stage is not None:
            if self._active_layer_id == _DEFAULT_LAYER_ID and not self._has_user_layers():
                self._reset_stage()
            else:
                self._remove_active_layer_prims()

        super().clear_model()

    @override
    def begin_frame(self, time: float):
        """
        Begin a new frame at the given simulation time.

        Args:
            time: The simulation time for the new frame.
        """
        super().begin_frame(time)
        self._frame_index = int(time * self.fps)
        self._frame_count += 1

        # Update stage end time if needed
        if self._frame_index > self.stage.GetEndTimeCode():
            self.stage.SetEndTimeCode(self._frame_index)

    @override
    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

    @override
    def is_running(self):
        """
        Check if the viewer is still running.

        Returns:
            bool: False if the frame limit is exceeded, True otherwise.
        """
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    @override
    def close(self):
        """
        Finalize and save the USD stage.

        This should be called when all logging is complete to ensure the USD file is written.
        """
        self.stage.GetRootLayer().Save()
        self.stage = None

        if self.output_path:
            print(f"USD output saved in: {os.path.abspath(self.output_path)}")

    def _get_path(self, name):
        # Handle both absolute and relative paths correctly
        if name.startswith("/"):
            return "/root" + name
        else:
            return "/root/" + name

    @override
    def log_mesh(
        self,
        name: str,
        points: wp.array[wp.vec3],
        indices: wp.array[wp.int32] | wp.array[wp.uint32],
        normals: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """
        Create a USD mesh prototype from vertex and index data.

        Args:
            name: Mesh name or Sdf.Path string.
            points: Vertex positions as a warp array of wp.vec3.
            indices: Triangle indices as a warp array of wp.uint32.
            normals: Vertex normals as a warp array of wp.vec3.
            uvs: UV coordinates as a warp array of wp.vec2.
            texture: Optional texture path/URL or image array.
            hidden: If True, mesh will be hidden.
            backface_culling: If True, enable backface culling.
            color: Optional base color as an RGB tuple with values in
                [0, 1]. Used when no texture is provided.
            roughness: Surface roughness in ``[0, 1]``. ``0`` is perfectly
                smooth, ``1`` is fully rough.
            metallic: Metallicity in ``[0, 1]``. ``0`` is dielectric, ``1``
                is metal.
        """

        name = self._qualify(name)

        # Convert warp arrays to numpy
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        # For hidden prototypes (geometry referenced by instances), store mesh data
        # in memory only — do NOT create a USD prim. Blender's USD importer renders
        # prototype prims even when marked invisible, so we avoid creating them.
        if hidden:
            if name not in self._mesh_data:
                normals_np = normals.numpy().astype(np.float32) if normals is not None else None
                uvs_np = uvs.numpy().astype(np.float32) if uvs is not None else None
                self._mesh_data[name] = {
                    "points": points_np,
                    "indices": indices_np,
                    "face_vertex_counts": [3] * (len(indices_np) // 3),
                    "normals": normals_np,
                    "uvs": uvs_np,
                    "num_verts": len(points_np),
                }
            return

        if name not in self._meshes:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            mesh_prim = UsdGeom.Mesh.Define(self.stage, self._get_path(name))

            # setup topology once (do not set every frame)
            face_vertex_counts = [3] * (len(indices_np) // 3)
            mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(indices_np)
            mesh_prim.GetSubdivisionSchemeAttr().Set("none")
            mesh_prim.GetDoubleSidedAttr().Set(True)

            # Set UVs once with topology (not time-sampled)
            if uvs is not None:
                uvs_np = uvs.numpy().astype(np.float32)
                primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
                uv_primvar = primvars_api.CreatePrimvar(
                    "st",
                    Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.vertex if len(uvs_np) == len(points_np) else UsdGeom.Tokens.faceVarying,
                )
                uv_primvar.Set(uvs_np)

            # Store the mesh prim
            self._meshes[name] = mesh_prim

            # Bind material for deformable meshes (warm cloth color matching GL viewer)
            if "/model/deformable/" in name:
                color = (0.7, 0.6, 0.4)
                mat_path = self._create_material(color, roughness=0.8, metallic=0.0)
                binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
                binding_api.Bind(UsdShade.Material(self.stage.GetPrimAtPath(mat_path)))

        mesh_prim = self._meshes[name]
        mesh_prim.GetPointsAttr().Set(points_np, self._frame_index)

        # Set normals if provided
        if normals is not None:
            normals_np = normals.numpy().astype(np.float32)
            mesh_prim.GetNormalsAttr().Set(normals_np, self._frame_index)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UVs if provided
        if uvs is not None:
            uvs_np = uvs.numpy().astype(np.float32) if isinstance(uvs, wp.array) else np.asarray(uvs, dtype=np.float32)
            pv_api = UsdGeom.PrimvarsAPI(mesh_prim)
            st_pv = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st_pv.Set(uvs_np)

        # Create and bind a textured material only when both texture and UVs are
        # provided — a UsdUVTexture shader with no "st" primvar would sample
        # undefined data and produce incorrect results.
        if texture is not None and uvs is not None and name not in self._texture_materials:
            self._create_texture_material(name, mesh_prim, texture)

        # Time-sampled visibility for animated meshes (deformable, toggling visibility per frame)
        mesh_prim.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    def _create_texture_material(self, mesh_name: str, mesh_prim, texture):
        """Create a UsdPreviewSurface material with a diffuse texture and bind it to *mesh_prim*."""
        from pxr import Sdf as _Sdf
        from pxr import UsdShade

        from ..utils.texture import load_texture  # noqa: PLC0415

        # Resolve texture to a file path on disk
        if isinstance(texture, str):
            tex_path = os.path.abspath(texture)
        else:
            tex_array = load_texture(texture)
            if tex_array is None:
                return
            tex_dir = os.path.dirname(self.output_path)
            safe_name = mesh_name.replace("/", "_").replace("\\", "_")
            tex_path = os.path.join(tex_dir, f"_tex_{safe_name}.png")
            try:
                self._save_texture_atomic(tex_array, tex_path)
            except Exception as exc:
                warnings.warn(
                    f"ViewerUSD: failed to export texture for mesh '{mesh_name}': {exc}. "
                    "Mesh will render without texture.",
                    stacklevel=2,
                )
                return

        mat_path = self._texture_material_path(mesh_name)
        self._ensure_scopes_for_path(self.stage, mat_path)

        material = UsdShade.Material.Define(self.stage, mat_path)
        surface = UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        diff_input = surface.CreateInput("diffuseColor", _Sdf.ValueTypeNames.Color3f)
        surface.CreateInput("roughness", _Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

        tex_reader = UsdShade.Shader.Define(self.stage, f"{mat_path}/DiffuseTexture")
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput("file", _Sdf.ValueTypeNames.Asset).Set(tex_path)
        tex_reader.CreateInput("sourceColorSpace", _Sdf.ValueTypeNames.Token).Set("auto")
        tex_reader.CreateInput("wrapS", _Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateInput("wrapT", _Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateOutput("rgb", _Sdf.ValueTypeNames.Float3)
        diff_input.ConnectToSource(tex_reader.ConnectableAPI(), "rgb")

        st_reader = UsdShade.Shader.Define(self.stage, f"{mat_path}/PrimvarSt")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", _Sdf.ValueTypeNames.Token).Set("st")
        st_reader.CreateOutput("result", _Sdf.ValueTypeNames.Float2)
        tex_reader.CreateInput("st", _Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")

        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
        UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(material)
        self._texture_materials[mesh_name] = material

    @staticmethod
    def _texture_material_path(mesh_name: str) -> str:
        safe = mesh_name.replace("/", "_").lstrip("_")
        return f"/root/Materials/mat_{safe}"

    # log a set of instances as individual mesh prims, slower but makes it easier
    # to do post-editing of instance materials etc. default for Newton shapes
    @override
    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | None,
        colors: wp.array[wp.vec3] | None,
        materials: wp.array[wp.vec4] | None,
        hidden: bool = False,
    ):
        """
        Log a batch of mesh instances for rendering.

        Args:
            name: Unique name for the instancer.
            mesh: Name of the base mesh.
            xforms: Array of transforms.
            scales: Array of scales.
            colors: Array of colors.
            materials: Array of materials.
            hidden: Whether the instances are hidden.
        """
        name = self._qualify(name)
        mesh = self._qualify(mesh)

        # Get mesh topology data (stored in memory, not as a USD prim). Instances define their
        # own mesh data rather than referencing a prototype: Blender's USD importer does not
        # reliably respect visibility on instance prims with internal references.
        if mesh not in self._mesh_data:
            msg = f"Mesh data '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        self._ensure_scopes_for_path(self.stage, self._get_path(name) + "/scope")
        group_prim = self.stage.GetPrimAtPath(self._get_path(name))
        if group_prim:
            self._instance_groups[name] = group_prim
            UsdGeom.Imageable(group_prim).GetVisibilityAttr().Set(
                "inherited" if not hidden else "invisible", self._frame_index
            )

        if xforms is not None:
            xforms = xforms.numpy()
        else:
            xforms = np.empty((0, 7), dtype=np.float32)

        if scales is not None:
            scales = scales.numpy()
        else:
            scales = np.ones((len(xforms), 3), dtype=np.float32)

        if colors is not None:
            colors = colors.numpy()

        mesh_info = self._mesh_data[mesh]

        for i in range(len(xforms)):
            instance_path = self._get_path(name) + f"/instance_{i}"
            instance = self.stage.GetPrimAtPath(instance_path)

            if not instance:
                # Define mesh data directly on each instance (no prototype references)
                mesh_prim = UsdGeom.Mesh.Define(self.stage, instance_path)
                mesh_prim.GetFaceVertexCountsAttr().Set(mesh_info["face_vertex_counts"])
                mesh_prim.GetFaceVertexIndicesAttr().Set(mesh_info["indices"])
                mesh_prim.GetPointsAttr().Set(mesh_info["points"])

                # Disable subdivision — Blender applies Catmull-Clark by default
                # which distorts meshes (especially low-poly collision proxies).
                mesh_prim.GetSubdivisionSchemeAttr().Set("none")
                # Render both sides — URDF .dae meshes often have inconsistent face
                # winding, and Blender's backface culling creates holes without this.
                mesh_prim.GetDoubleSidedAttr().Set(True)

                if mesh_info["uvs"] is not None:
                    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
                    uv_primvar = primvars_api.CreatePrimvar(
                        "st",
                        Sdf.ValueTypeNames.TexCoord2fArray,
                        UsdGeom.Tokens.vertex
                        if len(mesh_info["uvs"]) == mesh_info["num_verts"]
                        else UsdGeom.Tokens.faceVarying,
                    )
                    uv_primvar.Set(mesh_info["uvs"])

                instance = mesh_prim.GetPrim()
                _usd_add_xform(instance)

                # Bind material to instance
                if mesh in self._mesh_materials:
                    mat_color, mat_roughness, mat_metallic, mat_texture = self._mesh_materials[mesh]
                    inst_color = tuple(float(c) for c in colors[i]) if colors is not None else mat_color
                    mat_path = self._create_material(inst_color, mat_roughness, mat_metallic, mat_texture)
                    binding_api = UsdShade.MaterialBindingAPI.Apply(instance)
                    binding_api.Bind(UsdShade.Material(self.stage.GetPrimAtPath(mat_path)))

            UsdGeom.Imageable(instance).GetVisibilityAttr().Set(
                "inherited" if not hidden else "invisible", self._frame_index
            )

            # update transform
            if xforms is not None:
                pos = xforms[i][:3]
                rot = xforms[i][3:7]

                _usd_set_xform(instance, pos, rot, scales[i], self._frame_index)

            # update color
            if colors is not None:
                displayColor = UsdGeom.PrimvarsAPI(instance).GetPrimvar("displayColor")
                displayColor.Set(colors[i], self._frame_index)

    # log a set of instances as a point instancer, faster but less flexible
    def log_instances_point_instancer(
        self,
        name: str,
        mesh: str,
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | np.ndarray | None,
        colors: (
            wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | np.ndarray | None
        ),
        materials: wp.array[wp.vec4] | None,
    ):
        """
        Create or update a PointInstancer for mesh instances.

        Args:
            name: Instancer name or Sdf.Path string.
            mesh: Mesh prototype name (must be previously logged).
            xforms: Instance transforms as a warp array of wp.transform.
            scales: Instance scales as a warp array of wp.vec3.
            colors: Instance colors as a warp array of wp.vec3.
            materials: Instance materials as a warp array of wp.vec4.

        Raises:
            RuntimeError: If the mesh prototype is not found.
        """
        name = self._qualify(name)
        mesh = self._qualify(mesh)

        # Get prototype path
        if mesh not in self._meshes and mesh not in self._mesh_data:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        num_instances = len(xforms)

        # Create instancer if it doesn't exist
        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))
            instancer.CreateIdsAttr().Set(list(range(num_instances)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_instances)
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            # Set the prototype relationship
            instancer.GetPrototypesRel().AddTarget(self._get_path(mesh))

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        # Convert transforms to USD format
        if xforms is not None:
            xforms_np = xforms.numpy()

            # Extract positions from warp transforms using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            positions = xforms_np[:, :3].astype(np.float32)

            # Convert quaternion format: Warp (x, y, z, w) → USD (w, (x,y,z))
            # USD expects quaternions as Gf.Quath(real, imag_vec3)
            quat_w = xforms_np[:, 6].astype(np.float32)
            quat_xyz = xforms_np[:, 3:6].astype(np.float32)

            # Create orientations list with proper USD quaternion format
            orientations = []
            for i in range(num_instances):
                quat = Gf.Quath(
                    float(quat_w[i]), Gf.Vec3h(float(quat_xyz[i, 0]), float(quat_xyz[i, 1]), float(quat_xyz[i, 2]))
                )
                orientations.append(quat)

            # Handle scales with numpy operations
            if scales is None:
                scales = np.ones((num_instances, 3), dtype=np.float32)
            elif isinstance(scales, wp.array):
                scales = scales.numpy().astype(np.float32)

            # Set attributes at current time
            instancer.GetPositionsAttr().Set(positions, self._frame_index)
            instancer.GetOrientationsAttr().Set(orientations, self._frame_index)

            if scales is not None:
                instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                # Promote colors to proper numpy array format
                colors_np = self._promote_colors_to_array(colors, num_instances)

                # Set color per-instance
                displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                displayColor.Set(colors_np, self._frame_index)

                # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                indices = Vt.IntArray(range(num_instances))
                displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited", self._frame_index)

    # Abstract methods that need basic implementations
    @override
    def log_lines(
        self,
        name: str,
        starts: wp.array[wp.vec3] | None,
        ends: wp.array[wp.vec3] | None,
        colors: (wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None),
        width: float = 0.01,
        hidden: bool = False,
    ):
        """Debug helper to add a line list as a set of capsules

        Args:
            name: Unique name for the line batch.
            starts: The vertices of the lines (wp.array)
            ends: The vertices of the lines (wp.array)
            colors: The colors of the lines (wp.array)
            width: The width of the lines.
            hidden: Whether the lines are hidden.
        """

        # Skip hidden or empty debug visualizations — Blender ignores USD visibility flags
        if hidden or starts is None or ends is None:
            return

        name = self._qualify(name)

        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))

            # define nested capsule prim
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(width)

            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        if starts is not None and ends is not None:
            num_lines = int(len(starts))
            if num_lines > 0:
                # bring to host
                starts = starts.numpy()
                ends = ends.numpy()

                line_positions = []
                line_rotations = []
                line_scales = []

                for i in range(num_lines):
                    pos0 = starts[i]
                    pos1 = ends[i]

                    (pos, rot, scale) = _compute_segment_xform(
                        Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                        Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
                    )

                    line_positions.append(pos)
                    line_rotations.append(rot)
                    line_scales.append(scale)

                instancer.GetPositionsAttr().Set(line_positions, self._frame_index)
                instancer.GetOrientationsAttr().Set(line_rotations, self._frame_index)
                instancer.GetScalesAttr().Set(line_scales, self._frame_index)
                instancer.GetProtoIndicesAttr().Set([0] * num_lines, self._frame_index)
                instancer.CreateIdsAttr().Set(list(range(num_lines)))

                if colors is not None:
                    # Promote colors to proper numpy array format
                    colors_np = self._promote_colors_to_array(colors, num_lines)

                    # Set color per-instance
                    displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                    displayColor.Set(colors_np, self._frame_index)

                    # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                    indices = Vt.IntArray(range(num_lines))
                    displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    @override
    def log_points(
        self,
        name: str,
        points: wp.array[wp.vec3] | None,
        radii: wp.array[wp.float32] | float | None = None,
        colors: (wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None) = None,
        hidden: bool = False,
    ):
        """Log points as a USD primitive.

        By default, each point is an instance of a ``UsdGeom.Sphere`` prototype
        under a ``UsdGeom.PointInstancer``. Set ``points_as_spheres=False`` to
        write flat ``UsdGeom.Points`` splats instead.

        Args:
            name: Unique name for the point primitive.
            points: Point positions.
            radii: Point radii or a single shared radius.
            colors: Optional per-point colors or a shared RGB triplet.
            hidden: Whether the point primitive is hidden.

        Returns:
            ``Sdf.Path`` of the created/updated primitive.
        """
        name = self._qualify(name)

        if points is None:
            path = self._get_path(name)
            if self.points_as_spheres:
                instancer = UsdGeom.PointInstancer.Get(self.stage, path)
            else:
                instancer = UsdGeom.Points.Get(self.stage, path)
            if instancer:
                instancer.GetVisibilityAttr().Set("invisible", self._frame_index)
                return instancer.GetPath()
            return

        num_points = len(points)

        if radii is None:
            radii = 0.1

        if np.isscalar(radii):
            radius_interp = "constant"
        else:
            radius_interp = "vertex"

        colors, color_interp = self._normalize_point_colors(colors, num_points)

        path = self._get_path(name)

        if self.points_as_spheres:
            if name not in self._instancers:
                self._ensure_scopes_for_path(self.stage, path)
                instancer = UsdGeom.PointInstancer.Define(self.stage, path)
                sphere = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild("sphere"))
                sphere.GetRadiusAttr().Set(1.0)
                instancer.CreatePrototypesRel().SetTargets((sphere.GetPath(),))
                if colors is not None:
                    UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                        "displayColor", Sdf.ValueTypeNames.Color3fArray, color_interp, 1
                    )
                self._instancers[name] = instancer

            instancer = self._instancers[name]

            # Proto indices must be updated every frame: particle count can vary due to stream compaction.
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * num_points), self._frame_index)
            instancer.GetPositionsAttr().Set(points.numpy(), self._frame_index)

            # PointInstancer scales are vec3; broadcast scalar or per-point radius to (N, 3).
            if np.isscalar(radii):
                scales = np.full((num_points, 3), radii, dtype=np.float32)
            else:
                r = radii.numpy() if isinstance(radii, wp.array) else np.array(radii, dtype=np.float32)
                scales = np.stack([r, r, r], axis=1)
            instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                primvar = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                if not primvar:
                    primvar = UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                        "displayColor", Sdf.ValueTypeNames.Color3fArray, color_interp, 1
                    )
                primvar.Set(colors, self._frame_index)

            instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
            return instancer.GetPath()

        instancer = UsdGeom.Points.Get(self.stage, path)
        if not instancer:
            self._ensure_scopes_for_path(self.stage, path)
            instancer = UsdGeom.Points.Define(self.stage, path)
            self._points[name] = instancer

            UsdGeom.Primvar(instancer.GetWidthsAttr()).SetInterpolation(radius_interp)
            UsdGeom.Primvar(instancer.GetDisplayColorAttr()).SetInterpolation(color_interp)
        else:
            self._points[name] = instancer

        instancer.GetPointsAttr().Set(points.numpy(), self._frame_index)

        # convert radii to widths for USD
        if np.isscalar(radii):
            widths = (radii * 2.0,)
        elif isinstance(radii, wp.array):
            widths = radii.numpy() * 2.0
        else:
            widths = np.array(radii) * 2.0

        instancer.GetWidthsAttr().Set(widths, self._frame_index)

        if colors is not None:
            instancer.GetDisplayColorAttr().Set(colors, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
        return instancer.GetPath()

    @override
    def log_particle_spheres(self, name, points, radii, colors, hidden=False):
        """Render particles as instanced spheres using UsdGeom.PointInstancer.

        Args:
            name: Identifier for the particle instancer.
            points: Particle positions (wp.array of vec3).
            radii: Particle radii (wp.array of float).
            colors: Particle colors (wp.array of vec3 or None).
            hidden: Whether the particles should be hidden.
        """
        path = self._get_path(name)

        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, path)

            instancer = UsdGeom.PointInstancer.Define(self.stage, path)

            # Create sphere prototype
            sphere_path = path + "/proto_sphere"
            sphere = UsdGeom.Sphere.Define(self.stage, sphere_path)
            sphere.GetRadiusAttr().Set(1.0)

            instancer.CreatePrototypesRel().SetTargets([sphere.GetPath()])
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        if points is not None:
            points_np = points.numpy().astype(np.float32)
            num_particles = len(points_np)

            instancer.GetPositionsAttr().Set(points_np, self._frame_index)
            instancer.GetProtoIndicesAttr().Set([0] * num_particles, self._frame_index)
            instancer.CreateIdsAttr().Set(list(range(num_particles)))

            # Set scales from radii
            if isinstance(radii, wp.array):
                radii_np = radii.numpy().astype(np.float32)
            elif np.isscalar(radii):
                radii_np = np.full(num_particles, float(radii), dtype=np.float32)
            else:
                radii_np = np.array(radii, dtype=np.float32)

            scales = np.column_stack([radii_np, radii_np, radii_np])
            instancer.GetScalesAttr().Set(scales, self._frame_index)

            # Identity orientations
            orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * num_particles
            instancer.GetOrientationsAttr().Set(orientations, self._frame_index)

            if colors is not None:
                colors_np = self._promote_colors_to_array(colors, num_particles)
                displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                displayColor.Set(colors_np, self._frame_index)
                indices = Vt.IntArray(range(num_particles))
                displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    @override
    def log_array(self, name: str, array: wp.array[Any] | np.ndarray):
        """
        Log array data (not implemented for USD backend).

        This method is a placeholder and does not log array data in the USD backend.

        Args:
            name: Unique path/name for the array signal.
            array: Array data to visualize.
        """
        pass

    @override
    def log_scalar(self, name: str, value: int | float | bool | np.number, *, clear: bool = False, smoothing: int = 1):
        """
        Log scalar value (not implemented for USD backend).

        This method is a placeholder and does not log scalar values in the USD backend.

        Args:
            name: Unique path/name for the scalar signal.
            value: Scalar value to visualize.
            clear: Ignored by this backend.
            smoothing: Ignored by this backend.
        """
        pass

    @override
    def apply_forces(self, state: newton.State):
        """USD backend does not apply interactive forces.

        Args:
            state: Current simulation state.
        """
        pass

    def _create_material(self, color, roughness=0.5, metallic=0.0, texture=None):
        """Create or reuse a UsdPreviewSurface material.

        Args:
            color: RGB color tuple/list (0-1 range).
            roughness: Surface roughness (0-1).
            metallic: Metallic value (0-1).
            texture: Texture file path string, numpy image array, or None.

        Returns:
            str: USD path of the created/cached material.
        """
        color_tuple = (float(color[0]), float(color[1]), float(color[2]))
        texture_key = None
        if texture is not None:
            if isinstance(texture, str):
                texture_key = texture
            elif isinstance(texture, np.ndarray):
                texture_key = hash(texture.tobytes())
            else:
                texture_key = id(texture)

        mat_key = (color_tuple, float(roughness), float(metallic), texture_key)
        if mat_key in self._materials:
            return self._materials[mat_key]

        mat_idx = len(self._materials)
        mat_path = f"/root/materials/material_{mat_idx}"
        # Do NOT use _ensure_scopes_for_path here — it creates a "Scope" prim
        # at /root/materials which Blender imports as an Empty object, preventing
        # material discovery. UsdShade.Material.Define() handles parent creation.

        material = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/preview_surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))

        diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)

        if texture is not None:
            # Export texture file and create texture shader
            tex_rel_path = self._export_texture(texture)
            if tex_rel_path is not None:
                tex_shader = UsdShade.Shader.Define(self.stage, mat_path + "/texture")
                tex_shader.CreateIdAttr("UsdUVTexture")
                tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_rel_path)
                tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tex_output = tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

                # Primvar reader for UVs
                pv_reader = UsdShade.Shader.Define(self.stage, mat_path + "/primvar_reader")
                pv_reader.CreateIdAttr("UsdPrimvarReader_float2")
                pv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                pv_output = pv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

                tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(pv_output)
                diffuse_input.ConnectToSource(tex_output)
            else:
                diffuse_input.Set(Gf.Vec3f(*color_tuple))
        else:
            diffuse_input.Set(Gf.Vec3f(*color_tuple))

        surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface_output)

        self._materials[mat_key] = mat_path
        return mat_path

    def _export_texture(self, texture):
        """Export a texture to the textures/ subdirectory next to the USD output file.

        Args:
            texture: File path string or numpy image array.

        Returns:
            str: Relative asset path for USD reference (e.g., "./textures/texture_0.png"),
                 or None if export failed.
        """
        import shutil  # noqa: PLC0415

        if self._texture_dir is None:
            output_dir = os.path.dirname(os.path.abspath(self.output_path))
            self._texture_dir = os.path.join(output_dir, "textures")
            os.makedirs(self._texture_dir, exist_ok=True)

        tex_idx = len(os.listdir(self._texture_dir))

        if isinstance(texture, str):
            # Copy existing file
            if os.path.isfile(texture):
                ext = os.path.splitext(texture)[1] or ".png"
                dest_name = f"texture_{tex_idx}{ext}"
                dest_path = os.path.join(self._texture_dir, dest_name)
                shutil.copy2(texture, dest_path)
                return f"./textures/{dest_name}"
            return None
        elif isinstance(texture, np.ndarray):
            # Write numpy array as raw PPM (avoids PIL dependency)
            dest_name = f"texture_{tex_idx}.ppm"
            dest_path = os.path.join(self._texture_dir, dest_name)
            img = texture
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            h, w = img.shape[:2]
            with open(dest_path, "wb") as f:
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(img.tobytes())
            return f"./textures/{dest_name}"
        return None

    @override
    def log_material_info(self, mesh_name, color, roughness, metallic, texture):
        """Store material info and eagerly create the USD material.

        Materials must be defined in the USD before the meshes that reference them,
        because Blender's USD importer does not resolve forward references to materials.

        Args:
            mesh_name: The mesh prototype name.
            color: RGB color (vec3 or tuple).
            roughness: Surface roughness.
            metallic: Metallic value.
            texture: Texture data or None.
        """
        self._mesh_materials[mesh_name] = (color, roughness, metallic, texture)
        # Pre-create the material so it appears early in the USD file
        color_tuple = tuple(float(c) for c in color[:3])
        self._create_material(color_tuple, roughness, metallic, texture)

    def _promote_colors_to_array(self, colors, num_items):
        """
        Helper method to promote colors to a numpy array format.

        Parameters:
            colors: Input colors in various formats (wp.array, list/tuple, np.ndarray)
            num_items: Number of items that need colors

        Returns:
            np.ndarray: Colors as numpy array with shape (num_items, 3)
        """
        if colors is None:
            return None

        if isinstance(colors, wp.array):
            # Convert warp array to numpy
            return colors.numpy()
        elif isinstance(colors, list | tuple) and len(colors) == 3 and all(np.isscalar(x) for x in colors):
            # Single color (list/tuple of 3 floats) - promote to array with one value per item
            return np.tile(colors, (num_items, 1))
        elif isinstance(colors, np.ndarray):
            # Already numpy array - pass through
            return colors
        else:
            # Fallback for other formats
            return np.array(colors)

    @staticmethod
    def _is_single_rgb_triplet(colors) -> bool:
        """Returns True when colors represent one RGB triplet."""
        if isinstance(colors, np.ndarray):
            return colors.ndim == 1 and colors.shape[0] == 3

        if isinstance(colors, list | tuple):
            return len(colors) == 3 and all(np.isscalar(x) for x in colors)

        return False

    def _normalize_point_colors(self, colors, num_points):
        """Normalize point colors and return (values, interpolation token)."""
        if colors is None:
            return None, UsdGeom.Tokens.constant

        if isinstance(colors, wp.array):
            colors = colors.numpy()

        if self._is_single_rgb_triplet(colors):
            colors_arr = np.asarray(colors, dtype=np.float32)
            return colors_arr.reshape(1, 3), UsdGeom.Tokens.constant

        if isinstance(colors, np.ndarray):
            return colors, UsdGeom.Tokens.vertex

        if isinstance(colors, list | tuple):
            # Keep list/tuple inputs as-is for existing valid per-point color inputs.
            if len(colors) == num_points:
                return colors, UsdGeom.Tokens.vertex
            return np.asarray(colors), UsdGeom.Tokens.vertex

        return np.asarray(colors), UsdGeom.Tokens.vertex

    @staticmethod
    def _ensure_scopes_for_path(stage: Usd.Stage, prim_path_str: str):
        """
        Ensure that all parent prims in the hierarchy exist as 'Scope' prims.

        If a prim does not exist at the given path, this method creates all
        non-existent parent prims in its hierarchy as 'Scope' prims. This is
        useful for ensuring a valid hierarchy before defining a prim.

        Parameters:
            stage: The USD stage to operate on.
            prim_path_str: The Sdf.Path string for the target prim.
        """
        # Convert the string to an Sdf.Path object for robust manipulation
        prim_path = Sdf.Path(prim_path_str)

        # First, check if the target prim already exists.
        if stage.GetPrimAtPath(prim_path):
            return

        # We only want to create the parent hierarchy, not the final prim itself.
        parent_path = prim_path.GetParentPath()

        # GetPrefixes() provides a convenient list of all ancestor paths.
        # For "/A/B/C", it returns ["/", "/A", "/A/B"].
        for path in parent_path.GetPrefixes():
            # The absolute root path ('/') always exists, so we can skip it.
            if path == Sdf.Path.absoluteRootPath:
                continue

            # Check if a prim exists at the current ancestor path.
            if not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Scope")
