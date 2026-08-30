// Copyright 2026 Lars Brubaker
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Test 23 of src/manifold_tests/complex.rs — BooleanComplex.InterpolatedNormals —
// together with the two meshes it consumes.
//
// ── Provenance of the data ───────────────────────────────────────────────────
// The Rust does not spell these meshes out. It calls mod.rs's
// `read_cpp_test_source("boolean_complex_test.cpp")` and pulls the four `a.*`
// and four `b.*` initializer lists out of the pinned C++ test source AT TEST
// TIME. Per docs/PORTING_PLAN.md's verification note this port keeps no
// cpp-reference dependency, so the lists are transcribed here instead. Source:
//
//   manifold/test/boolean_complex_test.cpp, TEST(BooleanComplex,
//   InterpolatedNormals) — the eight assignments at lines 516–843 of commit
//   11235e6b8ebea2dbed8aec4285685aafd3d95667 (elalish/manifold v3.5.2), the
//   commit manifold-rust pins its `cpp-reference/manifold` submodule to.
//
// Transcribed mechanically by the same parse `cpp_inline_array` performs (strip
// `//` line comments, split on commas, parse each token), then verified
// token-for-token against the C++ — the MergeRefineData precedent.
//
// ── Why the decimals differ from the C++ text ────────────────────────────────
// The Rust parses each C++ token to f64 and then narrows: `.map(|v| v as f32)`.
// So the value a test actually sees is an f32, and its exact bits are what has
// to be reproduced — not the C++ decimal spelling. Each literal below is
// therefore the SHORTEST decimal that round-trips to that exact f32 (C#'s float
// literal parse is correctly rounded, so it lands on the same bits), which is
// why `-409.0570983886719` appears here as `-409.0571f`. Do not "restore" the
// longer C++ spellings: that would reintroduce a decimal→f64→f32 double
// rounding this transcription has already resolved.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldComplexTests
	{
		/// <summary>
		/// C++ TEST(BooleanComplex, InterpolatedNormals) — subtract two textured,
		/// normal-carrying meshes and verify the output maps back to the originals
		/// (RelatedGL with checkNormals + updateNormals).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInterpolatedNormals()
		{
			MeshGL a = new MeshGL();
			a.NumProp = 8;
			a.VertProperties = new List<float>(InterpNormalsAVertProperties);
			a.TriVerts = new List<uint>(InterpNormalsATriVerts);
			a.MergeFromVert = new List<uint>(InterpNormalsAMergeFromVert);
			a.MergeToVert = new List<uint>(InterpNormalsAMergeToVert);

			MeshGL b = new MeshGL();
			b.NumProp = 8;
			b.VertProperties = new List<float>(InterpNormalsBVertProperties);
			b.TriVerts = new List<uint>(InterpNormalsBTriVerts);
			b.MergeFromVert = new List<uint>(InterpNormalsBMergeFromVert);
			b.MergeToVert = new List<uint>(InterpNormalsBMergeToVert);

			a.RunOriginalId = new List<uint> { Manifold.ReserveIds(1) };
			b.RunOriginalId = new List<uint> { Manifold.ReserveIds(1) };

			Manifold aManifold = Manifold.FromMeshGL(a);
			Manifold bManifold = Manifold.FromMeshGL(b);

			Manifold aMinusB = aManifold - bManifold;

			await ManifoldTestHelpers.RelatedGlCheckNormals(aMinusB, new List<MeshGL> { a, b });
		}

		/// <summary>C++ <c>a.vertProperties</c> — 84 vertices at numProp 8 (xyz, normal xyz, uv).</summary>
		private static readonly float[] InterpNormalsAVertProperties =
		{
			-409.0571f, -300f, -198.83624f, 0f, -1f, 0f, 590.94293f, 301.16373f,
			-1000f, -300f, 500f, 0f, -1f, 0f, 0f, 1000f,
			-1000f, -300f, -500f, 0f, -1f, 0f, 0f, 0f,
			-1000f, -300f, -500f, -1f, 0f, 0f, 600f, 0f,
			-1000f, -300f, 500f, -1f, 0f, 0f, 600f, 1000f,
			-1000f, 300f, -500f, -1f, 0f, 0f, 0f, 0f,
			7.179657f, -300f, -330.03717f, 0f, -1f, 0f, 1007.1796f, 169.96284f,
			1000f, 300f, 500f, 0f, 0f, 1f, 2000f, 600f,
			403.5837f, 300f, 500f, 0f, 0f, 1f, 1403.5837f, 600f,
			564.2904f, 21.64801f, 500f, 0f, 0f, 1f, 1564.2905f, 321.648f,
			1000f, -300f, -500f, 0f, 0f, -1f, 2000f, 600f,
			-1000f, -300f, -500f, 0f, 0f, -1f, 0f, 600f,
			-1000f, 300f, -500f, 0f, 0f, -1f, 0f, 0f,
			1000f, 300f, 500f, 0f, 1f, 0f, 0f, 1000f,
			1000f, 300f, -500f, 0f, 1f, 0f, 0f, 0f,
			724.52716f, 300f, 398.83624f, 0f, 1f, 0f, 275.47284f, 898.83624f,
			-115.352554f, -300f, 500f, 0f, -1f, 0f, 884.6475f, 1000.0001f,
			-384.7195f, 166.55722f, 500f, 0f, 0f, 1f, 615.2805f, 466.55725f,
			-1000f, -300f, 500f, 0f, 0f, 1f, 0f, 0f,
			-161.6137f, -219.87335f, 500f, 0f, 0f, 1f, 838.3863f, 80.12665f,
			1000f, -300f, 500f, 0f, 0f, 1f, 2000f, 0f,
			-115.352554f, -300f, 500f, 0f, 0f, 1f, 884.6475f, 0f,
			1000f, 300f, 500f, 1f, 0f, 0f, 600f, 1000f,
			1000f, -300f, 500f, 1f, 0f, 0f, 0f, 1000f,
			1000f, 300f, -500f, 1f, 0f, 0f, 600f, 0f,
			566.6258f, 300f, 23.128052f, 0f, 1f, 0f, 433.37424f, 523.1281f,
			411.5867f, -66.51549f, -500f, 0f, 0f, -1f, 1411.5867f, 366.5155f,
			375.74988f, -4.4443007f, -500f, 0f, 0f, -1f, 1375.7499f, 304.4443f,
			346.7673f, 300f, -500f, 0f, 1f, 0f, 653.2326f, 0f,
			-153.58984f, 300f, -388.5525f, 0f, 1f, 0f, 1153.5898f, 111.44751f,
			199.97888f, 300f, -500f, 0f, 1f, 0f, 800.0212f, 0f,
			-1000f, 300f, -500f, 0f, 1f, 0f, 2000f, 0f,
			-153.58987f, 300f, 44.222473f, 0f, 1f, 0f, 1153.5898f, 544.2225f,
			199.97888f, 300f, -500f, 0f, 0f, -1f, 1199.9791f, 0f,
			521.67804f, -2.954248f, -500f, 0f, 0f, -1f, 1521.678f, 302.95425f,
			346.7673f, 300f, -500f, 0f, 0f, -1f, 1346.7673f, 0f,
			1000f, 300f, -500f, 0f, 0f, -1f, 2000f, 0f,
			-1000f, 300f, 500f, -1f, 0f, 0f, 0f, 1000f,
			-1000f, 300f, 500f, 0f, 0f, 1f, 0f, 600f,
			-1000f, 300f, 500f, 0f, 1f, 0f, 2000f, 1000f,
			-153.58986f, 300f, 500f, 0f, 0f, 1f, 846.4102f, 600f,
			88.46628f, -253.06915f, 500f, 0f, 0f, 1f, 1088.4664f, 46.930847f,
			-153.58986f, 300f, 500f, 0f, 1f, 0f, 1153.5898f, 1000f,
			7.17967f, -300f, 500f, 0f, 0f, 1f, 1007.17975f, 0f,
			1000f, -300f, -500f, 0f, -1f, 0f, 2000f, 0f,
			1000f, -300f, 500f, 0f, -1f, 0f, 2000f, 1000f,
			7.17967f, -300f, 500f, 0f, -1f, 0f, 1007.1796f, 1000f,
			403.5837f, 300f, 500f, 0f, 1f, 0f, 596.4163f, 1000f,
			1000f, -300f, -500f, 1f, 0f, 0f, 0f, 0f,
			492.30057f, -19.915321f, -500f, 0f, 0f, -1f, 1492.3005f, 319.91534f,
			411.5867f, -66.51549f, -500f, -0.5f, 0.8660254f, 0f, 880.54395f, 0f,
			7.179657f, -300f, -330.03717f, -0.50000006f, 0.86602545f, 0f, 383.60587f, 0f,
			492.30057f, -19.915321f, -500f, -0.5f, 0.8660254f, 0f, 968.1236f, 31.876385f,
			7.17967f, -300f, 500f, -0.50000006f, 0.86602545f, 0f, 99.716446f, 779.97974f,
			88.46628f, -253.06915f, 500f, -0.5f, 0.8660254f, 0f, 187.91759f, 812.0824f,
			-153.58986f, 300f, 500f, 0.5f, -0.86602545f, 0f, 749.2096f, 834.9662f,
			-384.7195f, 166.55722f, 500f, 0.50000006f, -0.86602545f, 0f, 1000f, 743.686f,
			-153.58987f, 300f, 44.222473f, 0.5f, -0.8660254f, 0f, 593.3245f, 406.67545f,
			564.2904f, 21.64801f, 500f, -0.5f, 0.86602545f, 0f, 704.21704f, 1000.00006f,
			-604.9979f, 39.37943f, -198.83624f, 0.5f, -0.8660254f, 0f, 1000f, 0f,
			199.97888f, 300f, -500f, 0.29619816f, 0.17101009f, 0.93969274f, 880.5439f, 176.78435f,
			-153.58984f, 300f, -388.5525f, 0.29619816f, 0.17101009f, 0.93969274f, 554.69324f, 0f,
			375.74988f, -4.4443007f, -500f, 0.29619813f, 0.17101009f, 0.9396927f, 880.5439f, 528.32635f,
			566.6258f, 300f, 23.128052f, -0.8137977f, -0.46984637f, 0.34202018f, 239.89218f, 600.1797f,
			346.7673f, 300f, -500f, -0.8137978f, -0.46984634f, 0.34202018f, 349.8214f, 43.47846f,
			521.67804f, -2.954248f, -500f, -0.8137978f, -0.46984634f, 0.34202018f, 0f, 43.47846f,
			804.9979f, 160.62057f, 398.83624f, -0.5f, 0.8660254f, 0f, 1000f, 1000f,
			521.67804f, -2.954248f, -500f, -0.5f, 0.8660254f, 0f, 1000f, 43.47838f,
			-153.58984f, 300f, -388.5525f, 0.5f, -0.8660254f, 0f, 445.30676f, 0f,
			-604.9979f, 39.37943f, -198.83624f, 0.29619816f, 0.17101009f, 0.93969274f, 0f, 0f,
			804.9979f, 160.62057f, 398.83624f, -0.81379765f, -0.4698463f, 0.34202015f, 0f, 1000f,
			-161.6137f, -219.87335f, 500f, 0.8137977f, 0.4698463f, -0.34202015f, 446.2116f, 743.68604f,
			-604.9979f, 39.37943f, -198.83624f, 0.81379765f, 0.4698463f, -0.34202015f, 0f, 0f,
			-384.7195f, 166.55722f, 500f, 0.8137977f, 0.46984637f, -0.34202018f, 0f, 743.686f,
			-115.352554f, -300f, 500f, 0.81379765f, 0.46984634f, -0.34202015f, 538.7339f, 743.68604f,
			-409.0571f, -300f, -198.83624f, 0.81379765f, 0.4698463f, -0.34202015f, 391.88162f, 0f,
			7.179657f, -300f, -330.03717f, 0.29619816f, 0.17101009f, 0.93969274f, 383.60587f, 600f,
			564.2904f, 21.64801f, 500f, -0.29619813f, -0.17101009f, -0.9396926f, 704.2169f, -3.0517578e-05f,
			403.5837f, 300f, 500f, -0.29619813f, -0.17101009f, -0.9396926f, 704.217f, 321.41324f,
			724.52716f, 300f, 398.83624f, -0.29619816f, -0.17101009f, -0.9396927f, 1000f, 160.9415f,
			804.9979f, 160.62057f, 398.83624f, -0.29619816f, -0.17101009f, -0.93969274f, 1000f, 0f,
			-409.0571f, -300f, -198.83624f, 0.29619816f, 0.17101009f, 0.93969274f, 0f, 391.88165f,
			724.52716f, 300f, 398.83624f, -0.81379765f, -0.4698463f, 0.34202018f, 160.9415f, 1000.00006f,
			411.5867f, -66.51549f, -500f, 0.29619816f, 0.17101008f, 0.9396926f, 880.544f, 600.00006f,
		};

		/// <summary>C++ <c>a.triVerts</c> — 56 triangles.</summary>
		private static readonly uint[] InterpNormalsATriVerts =
		{
			0,   1,   2,   3,   4,   5,   6,   0,   2,   7,   8,   9,
			10,  11,  12,  13,  14,  15,  0,   16,  1,   17,  18,  19,
			9,   20,  7,   18,  21,  19,  22,  23,  24,  14,  25,  15,
			26,  12,  27,  14,  28,  25,  29,  30,  31,  29,  31,  32,
			12,  33,  27,  34,  35,  36,  5,   4,   37,  17,  38,  18,
			31,  39,  32,  40,  38,  17,  9,   41,  20,  39,  42,  32,
			41,  43,  20,  6,   2,   44,  6,   45,  46,  26,  10,  12,
			47,  13,  15,  48,  24,  23,  6,   44,  45,  26,  49,  10,
			49,  34,  10,  34,  36,  10,  50,  51,  52,  51,  53,  54,
			51,  54,  52,  55,  56,  57,  52,  54,  58,  59,  57,  56,
			60,  61,  62,  63,  64,  65,  52,  66,  67,  59,  68,  57,
			69,  62,  61,  65,  70,  63,  71,  72,  73,  52,  58,  66,
			74,  72,  71,  74,  75,  72,  62,  69,  76,  77,  78,  79,
			79,  80,  77,  69,  81,  76,  63,  70,  82,  76,  83,  62,
		};

		/// <summary>C++ <c>a.mergeFromVert</c>.</summary>
		private static readonly uint[] InterpNormalsAMergeFromVert =
		{
			3,   4,   11,  12,  13,  18,  21,  22,  23,  24,  31,  33,  35,  36,
			38,  39,  42,  44,  45,  46,  47,  48,  50,  51,  52,  53,  54,  55,
			56,  57,  58,  60,  61,  62,  63,  64,  65,  67,  68,  69,  70,  71,
			72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
		};

		/// <summary>C++ <c>a.mergeToVert</c>.</summary>
		private static readonly uint[] InterpNormalsAMergeToVert =
		{
			2,   1,   2,   5,   7,   1,   16,  7,   20,  14,  5,   30,  28,  14,
			37,  37,  40,  10,  20,  43,  8,   10,  26,  6,   49,  43,  41,  40,
			17,  32,  9,   30,  29,  27,  25,  28,  34,  34,  29,  59,  66,  19,
			59,  17,  16,  0,   6,   9,   8,   15,  66,  0,   15,  26,
		};

		/// <summary>C++ <c>b.vertProperties</c> — 24 vertices at numProp 8.</summary>
		private static readonly float[] InterpNormalsBVertProperties =
		{
			-1700f, -600f, -1000f, -1f, 0f, 0f, 1200f, 0f,
			-1700f, -600f, 1000f, -1f, 0f, 0f, 1200f, 2000f,
			-1700f, 600f, -1000f, -1f, 0f, 0f, 0f, 0f,
			-1700f, -600f, -1000f, 0f, -1f, 0f, 0f, 0f,
			300f, -600f, -1000f, 0f, -1f, 0f, 2000f, 0f,
			-1700f, -600f, 1000f, 0f, -1f, 0f, 0f, 2000f,
			-1700f, -600f, -1000f, 0f, 0f, -1f, 0f, 1200f,
			-1700f, 600f, -1000f, 0f, 0f, -1f, 0f, 0f,
			300f, -600f, -1000f, 0f, 0f, -1f, 2000f, 1200f,
			-1700f, -600f, 1000f, 0f, 0f, 1f, 0f, 0f,
			300f, -600f, 1000f, 0f, 0f, 1f, 2000f, 0f,
			-1700f, 600f, 1000f, 0f, 0f, 1f, 0f, 1200f,
			-1700f, 600f, 1000f, -1f, 0f, 0f, 0f, 2000f,
			-1700f, 600f, -1000f, 0f, 1f, 0f, 2000f, 0f,
			-1700f, 600f, 1000f, 0f, 1f, 0f, 2000f, 2000f,
			300f, 600f, 1000f, 0f, 1f, 0f, 0f, 2000f,
			300f, -600f, -1000f, 1f, 0f, 0f, 0f, 0f,
			300f, 600f, -1000f, 1f, 0f, 0f, 1200f, 0f,
			300f, -600f, 1000f, 1f, 0f, 0f, 0f, 2000f,
			300f, -600f, 1000f, 0f, -1f, 0f, 2000f, 2000f,
			300f, 600f, -1000f, 0f, 0f, -1f, 2000f, 0f,
			300f, 600f, -1000f, 0f, 1f, 0f, 0f, 0f,
			300f, 600f, 1000f, 0f, 0f, 1f, 2000f, 1200f,
			300f, 600f, 1000f, 1f, 0f, 0f, 1200f, 2000f,
		};

		/// <summary>C++ <c>b.triVerts</c> — 12 triangles.</summary>
		private static readonly uint[] InterpNormalsBTriVerts =
		{
			0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
			1,   12,  2,   13,  14,  15,  16,  17,  18,  4,   19,  5,
			7,   20,  8,   21,  13,  15,  10,  22,  11,  17,  23,  18,
		};

		/// <summary>C++ <c>b.mergeFromVert</c>.</summary>
		private static readonly uint[] InterpNormalsBMergeFromVert =
		{
			3,   5,   6,   7,   8,   9,   12,  13,  14,  16,  18,  19,  20,  21,  22,  23,
		};

		/// <summary>C++ <c>b.mergeToVert</c>.</summary>
		private static readonly uint[] InterpNormalsBMergeToVert =
		{
			0,   1,   0,   2,   4,   1,   11,  2,   11,  4,   10,  10,  17,  17,  15,  15,
		};
	}
}
