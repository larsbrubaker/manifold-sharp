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

// Port of cross_section_transform_tests.rs — all 8 cases, same inputs, same
// expected bit patterns, same order. They pin CrossSection's lazy affine
// transforms (Translate / Rotate / Scale / Mirror in CrossSection.cs) to the C++
// reference: the composed mat2x3, its application as `m * vec3(x, y, 1)`, the
// identity shortcut, winding reversal on a negative determinant, and the point at
// which a read materializes the transform. Expected bit patterns come from the
// C++ reference compiled with MSVC against Clipper2 46f6391.
//
// ReadersApplyPendingTransformLikeCpp's union runs through Clipper2Lib's engine,
// which docs/FOLLOW_UPS.md records as ordering ties differently from
// clipper2-rust; this input has no such tie and matches the Rust as written.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class CrossSectionTransformTests
	{
		private static readonly (ulong, ulong)[][] Chain =
		{
			new[]
			{
				(0x3ff120ab8f1b81cdUL, 0xbfe77420ce509266UL),
				(0x3fdb35858b1ea3f2UL, 0x3fc10e6311d12e36UL),
				(0xbfdf32889fab9bc0UL, 0x3ff07e32c3f80733UL),
				(0xbff21fb18c247aebUL, 0x3ff6f18ced3e5df7UL),
				(0xbff21f87fb86eff7UL, 0x3ff1b483e64af552UL),
				(0xbfdf30f73ccc5c9aUL, 0x3fcec538e74432c6UL),
				(0x3fdb3716edfde318UL, 0xbfe5077e89aab626UL),
				(0x3ff120d51fb90cc1UL, 0xbff0f7196e1bb1d7UL),
			},
		};

		private static readonly (ulong, ulong)[][] ChainMat =
		{
			new[]
			{
				(0x3ff120ab8f1b81cdUL, 0xbfe77420ce509267UL),
				(0x3fdb35858b1ea3f3UL, 0x3fc10e6311d12e36UL),
				(0xbfdf32889fab9bc1UL, 0x3ff07e32c3f80732UL),
				(0xbff21fb18c247aecUL, 0x3ff6f18ced3e5df7UL),
				(0xbff21f87fb86eff7UL, 0x3ff1b483e64af552UL),
				(0xbfdf30f73ccc5c9bUL, 0x3fcec538e74432c2UL),
				(0x3fdb3716edfde319UL, 0xbfe5077e89aab627UL),
				(0x3ff120d51fb90cc1UL, 0xbff0f7196e1bb1d7UL),
			},
		};

		private static readonly (ulong, ulong)[][] ThereBack =
		{
			new[]
			{
				(0x3ff0000000000000UL, 0x0000000000000000UL),
				(0x3fe6a09e667f3bccUL, 0x3fe6a09e667f3bccUL),
				(0x8000000000000000UL, 0x3ff0000000000000UL),
				(0xbfe6a09e667f3bccUL, 0x3fe6a09e667f3bccUL),
				(0xbff0000000000000UL, 0x8000000000000000UL),
				(0xbfe6a09e667f3bccUL, 0xbfe6a09e667f3bccUL),
				(0x0000000000000000UL, 0xbff0000000000000UL),
				(0x3fe6a09e667f3bccUL, 0xbfe6a09e667f3bccUL),
			},
		};

		private static readonly (ulong, ulong)[][] Rot90 =
		{
			new[]
			{
				(0x0000000000000000UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0x4000000000000000UL),
				(0xbff0000000000000UL, 0x4000000000000000UL),
				(0xbff0000000000000UL, 0x0000000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] Rot180 =
		{
			new[]
			{
				(0x0000000000000000UL, 0x0000000000000000UL),
				(0xc000000000000000UL, 0x0000000000000000UL),
				(0xc000000000000000UL, 0xbff0000000000000UL),
				(0x0000000000000000UL, 0xbff0000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] Rotm90 =
		{
			new[]
			{
				(0x0000000000000000UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0xc000000000000000UL),
				(0x3ff0000000000000UL, 0xc000000000000000UL),
				(0x3ff0000000000000UL, 0x0000000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] Rot45 =
		{
			new[]
			{
				(0x0000000000000000UL, 0x0000000000000000UL),
				(0x3ff6a09e667f3bccUL, 0x3ff6a09e667f3bccUL),
				(0x3fe6a09e667f3bccUL, 0x4000f876ccdf6cd9UL),
				(0xbfe6a09e667f3bccUL, 0x3fe6a09e667f3bccUL),
			},
		};

		private static readonly (ulong, ulong)[][] Rot30C8 =
		{
			new[]
			{
				(0x3febb67ae8584cabUL, 0x3fdfffffffffffffUL),
				(0x3fd0907dc1930691UL, 0x3feee8dd4748bf14UL),
				(0xbfdfffffffffffffUL, 0x3febb67ae8584cabUL),
				(0xbfeee8dd4748bf14UL, 0x3fd0907dc1930691UL),
				(0xbfebb67ae8584cabUL, 0xbfdfffffffffffffUL),
				(0xbfd0907dc1930691UL, 0xbfeee8dd4748bf14UL),
				(0x3fdfffffffffffffUL, 0xbfebb67ae8584cabUL),
				(0x3feee8dd4748bf14UL, 0xbfd0907dc1930691UL),
			},
		};

		private static readonly (ulong, ulong)[][] Mirror1E11 =
		{
			new[]
			{
				(0x0000000000000000UL, 0x3ff0000000000000UL),
				(0xc000000000000000UL, 0x3ff0000000000000UL),
				(0xc000000000000000UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0x0000000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] Mirror1E160 =
		{
			new[]
			{
				(0x0000000000000000UL, 0x3ff0000000000000UL),
				(0xc000001758f3cba8UL, 0x3ff0000000000000UL),
				(0xc000001758f3cba8UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0x0000000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] MirrorY =
		{
			new[]
			{
				(0x0000000000000000UL, 0xbff0000000000000UL),
				(0x4000000000000000UL, 0xbff0000000000000UL),
				(0x4000000000000000UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0x0000000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] ScaleNeg =
		{
			new[]
			{
				(0x0000000000000000UL, 0x3ff0000000000000UL),
				(0xc000000000000000UL, 0x3ff0000000000000UL),
				(0xc000000000000000UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0x0000000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] ScaleZero =
		{
			new[]
			{
				(0x0000000000000000UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0x0000000000000000UL),
				(0x0000000000000000UL, 0x3ff0000000000000UL),
				(0x0000000000000000UL, 0x3ff0000000000000UL),
			},
		};

		private static readonly (ulong, ulong)[][] UnionTransformed =
		{
			new[]
			{
				(0x3fe16daed8000000UL, 0xbfead663a8000000UL),
				(0x3fef4cfc34000000UL, 0xbfca9cd9b0000000UL),
				(0x3fec95bd30000000UL, 0x3fd0000000000000UL),
				(0x4004000000000000UL, 0x3fd0000000000000UL),
				(0x4004000000000000UL, 0x3ff4000000000000UL),
				(0x3fe0000000000000UL, 0x3ff4000000000000UL),
				(0x3fe0000000000000UL, 0x3fe8e077c8000000UL),
				(0x3fca9cd9b0000000UL, 0x3fef4cfc34000000UL),
				(0xbfe16daed8000000UL, 0x3fead663a8000000UL),
				(0xbfef4cfc34000000UL, 0x3fca9cd9b0000000UL),
				(0xbfead663a8000000UL, 0xbfe16daed8000000UL),
				(0xbfca9cd9b0000000UL, 0xbfef4cfc34000000UL),
			},
		};
		/// <summary>
		/// C++ composes every transform into one mat2x3 (<c>m * Mat3(transform_)</c>) and
		/// applies it once on read, so a chain rounds differently from applying each step
		/// to the vertices.
		/// </summary>
		[Test]
		public async Task ChainedTransformsComposeLazilyLikeCpp()
		{
			CrossSection cs = CrossSection.Circle(1.0, 8)
				.Translate(new Vec2(0.1, 0.2))
				.Rotate(33.0)
				.Scale(new Vec2(1.7, -0.3))
				.Mirror(new Vec2(0.3, 0.7))
				.Translate(new Vec2(-0.05, 0.11));
			await Assert.That(BitsEqual(cs.ToPolygons(), Chain)).IsTrue();
			await Assert.That(BitConverter.DoubleToUInt64Bits(cs.Area())).IsEqualTo(0x3ff714789bbf37deUL);
		}

		/// <summary>
		/// A read (C++ <c>GetPaths</c>) bakes the pending transform into the paths and
		/// resets it to identity, so later transforms compose from the baked paths.
		/// </summary>
		[Test]
		public async Task ReadMaterializesTransformLikeCpp()
		{
			CrossSection x = CrossSection.Circle(1.0, 8)
				.Translate(new Vec2(0.1, 0.2))
				.Rotate(33.0);
			_ = x.Area();
			CrossSection cs = x
				.Scale(new Vec2(1.7, -0.3))
				.Mirror(new Vec2(0.3, 0.7))
				.Translate(new Vec2(-0.05, 0.11));
			await Assert.That(BitsEqual(cs.ToPolygons(), ChainMat)).IsTrue();
			await Assert.That(BitConverter.DoubleToUInt64Bits(cs.Area())).IsEqualTo(0x3ff714789bbf37dfUL);
		}

		/// <summary>
		/// Translating there and back composes to exactly the identity, which C++
		/// <c>GetPaths</c> skips, so the vertices (including the circle's -0.0s) are
		/// returned untouched.
		/// </summary>
		[Test]
		public async Task IdentityCompositeLeavesPathsUntouched()
		{
			CrossSection cs = CrossSection.Circle(1.0, 8)
				.Translate(new Vec2(0.1, 0.3))
				.Translate(new Vec2(-0.1, -0.3));
			await Assert.That(BitsEqual(cs.ToPolygons(), ThereBack)).IsTrue();
		}

		/// <summary>
		/// C++ <c>Rotate</c> takes <c>sind</c> / <c>cosd</c>, which are exact at multiples
		/// of 90 degrees.
		/// </summary>
		[Test]
		public async Task RotateUsesSindCosdLikeCpp()
		{
			CrossSection r = CrossSection.SquareVec2(new Vec2(2.0, 1.0), false);
			await Assert.That(BitsEqual(r.Rotate(90.0).ToPolygons(), Rot90)).IsTrue();
			await Assert.That(BitsEqual(r.Rotate(180.0).ToPolygons(), Rot180)).IsTrue();
			await Assert.That(BitsEqual(r.Rotate(-90.0).ToPolygons(), Rotm90)).IsTrue();
			await Assert.That(BitsEqual(r.Rotate(45.0).ToPolygons(), Rot45)).IsTrue();
			await Assert.That(BitConverter.DoubleToUInt64Bits(r.Rotate(45.0).Area())).IsEqualTo(0x3ffffffffffffffeUL);
			CrossSection c = CrossSection.Circle(1.0, 8).Rotate(30.0);
			await Assert.That(BitsEqual(c.ToPolygons(), Rot30C8)).IsTrue();
		}

		/// <summary>
		/// C++ <c>Mirror</c> returns an empty section only when <c>la::length(ax) == 0</c>
		/// (which includes lengths that underflow); any other axis is normalized and
		/// reflected, with the winding reversed by the negative determinant.
		/// </summary>
		[Test]
		public async Task MirrorGuardAndMatrixLikeCpp()
		{
			CrossSection r = CrossSection.SquareVec2(new Vec2(2.0, 1.0), false);
			await Assert.That(r.Mirror(new Vec2(0.0, 0.0)).NumVert()).IsEqualTo(0);
			await Assert.That(BitsEqual(r.Mirror(new Vec2(1e-11, 0.0)).ToPolygons(), Mirror1E11)).IsTrue();
			await Assert.That(BitsEqual(r.Mirror(new Vec2(1e-160, 0.0)).ToPolygons(), Mirror1E160)).IsTrue();
			await Assert.That(r.Mirror(new Vec2(1e-170, 0.0)).NumVert()).IsEqualTo(0);
			await Assert.That(BitsEqual(r.Mirror(new Vec2(0.0, 1.0)).ToPolygons(), MirrorY)).IsTrue();
		}

		/// <summary>
		/// Scale goes through the same matrix: a negative determinant reverses each
		/// contour, a zero one does not.
		/// </summary>
		[Test]
		public async Task ScaleMatrixAndWindingLikeCpp()
		{
			CrossSection r = CrossSection.SquareVec2(new Vec2(2.0, 1.0), false);
			await Assert.That(BitsEqual(r.Scale(new Vec2(-1.0, 1.0)).ToPolygons(), ScaleNeg)).IsTrue();
			await Assert.That(BitsEqual(r.Scale(new Vec2(0.0, 1.0)).ToPolygons(), ScaleZero)).IsTrue();
		}

		/// <summary>
		/// <c>m * vec3(x, y, 1)</c> multiplies every column, so translating the infinite
		/// corners of <c>CrossSection.FromRect(new Rect())</c> gives <c>0 * inf</c> = NaN in
		/// every coordinate, as in C++. (NaN sign and payload are not specified by Rust, so
		/// only NaN-ness is compared.)
		/// </summary>
		[Test]
		public async Task TranslateInfiniteRectGivesNanLikeCpp()
		{
			CrossSection cs = CrossSection.FromRect(new Rect()).Translate(new Vec2(1.0, 2.0));
			Polygons p = cs.ToPolygons();
			await Assert.That(p.Count).IsEqualTo(1);
			await Assert.That(p[0].Count).IsEqualTo(4);
			await Assert.That(p[0].All(v => double.IsNaN(v.X) && double.IsNaN(v.Y))).IsTrue();
			await Assert.That(double.IsNaN(cs.Area())).IsTrue();
		}

		/// <summary>Bounds and booleans read through the pending transform too.</summary>
		[Test]
		public async Task ReadersApplyPendingTransformLikeCpp()
		{
			Rect b = CrossSection.Circle(1.0, 8)
				.Translate(new Vec2(0.1, 0.2))
				.Rotate(33.0)
				.Bounds();
			await Assert.That((
					BitConverter.DoubleToUInt64Bits(b.Min.X),
					BitConverter.DoubleToUInt64Bits(b.Min.Y),
					BitConverter.DoubleToUInt64Bits(b.Max.X),
					BitConverter.DoubleToUInt64Bits(b.Max.Y)))
				.IsEqualTo((0xbff00d243325f02bUL, 0xbfe830bd2e65849dUL, 0x3fee7faffea820a9UL, 0x3ff3349d9b473e31UL));
			CrossSection r = CrossSection.SquareVec2(new Vec2(2.0, 1.0), false);
			CrossSection u = CrossSection.Circle(1.0, 8)
				.Rotate(33.0)
				.Union(r.Translate(new Vec2(0.5, 0.25)));
			await Assert.That(BitsEqual(u.ToPolygons(), UnionTransformed)).IsTrue();
			await Assert.That(BitConverter.DoubleToUInt64Bits(u.Area())).IsEqualTo(0x4012b987c182097bUL);
		}

		/// <summary>
		/// The Rust <c>assert_eq!(bits(p), want(b))</c>: contour count, vertex count and
		/// every coordinate's bit pattern, in order.
		/// </summary>
		private static bool BitsEqual(Polygons p, (ulong X, ulong Y)[][] want)
		{
			if (p.Count != want.Length)
			{
				return false;
			}

			for (int i = 0; i < p.Count; i++)
			{
				if (p[i].Count != want[i].Length)
				{
					return false;
				}

				for (int j = 0; j < p[i].Count; j++)
				{
					if (BitConverter.DoubleToUInt64Bits(p[i][j].X) != want[i][j].X
						|| BitConverter.DoubleToUInt64Bits(p[i][j].Y) != want[i][j].Y)
					{
						return false;
					}
				}
			}

			return true;
		}
	}
}
