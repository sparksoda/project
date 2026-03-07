//bubbles.js
/*
* bubbles.js
* build in Soda SLM
* by @sparksoda,@Catmakura
* https://github.com/sparksoda/project/
*/
//変数の初期指定
const Engine = Matter.Engine,
      Render = Matter.Render,
      World = Matter.World,
      Bodies = Matter.Bodies,
      Body = Matter.Body,
      Runner = Matter.Runner;

const engine = Engine.create();
const world = engine.world;

const render = Render.create({
  element: document.body,
  engine: engine,
  options: {
    width: window.innerWidth,
    height: window.innerHeight,
    wireframes: false,
    background: 'transparent'
  }
});

Render.run(render);
const runner = Runner.create();
Runner.run(runner, engine);

function createBubble() {
  const radius = 5 + Math.random() * 15; 
  const opacity = 0.3 + (radius - 5) / 15 * 0.7;
  const buoyancy = 0.0003 + (radius - 5) / 15 * 0.0007;

  const bubble = Bodies.circle(
    Math.random() * window.innerWidth,
    window.innerHeight + radius,
    radius,
    {
      restitution: 0.8,
      frictionAir: 0.01,  // 少し軽く
      render: {
        fillStyle: `rgba(255,255,255,${opacity})`
      }
    }
  );

  // 安全のため custom プロパティを必ず作成
  bubble.custom = { buoyancy: buoyancy };

  World.add(world, bubble);
  return bubble;
}

const bubbles = [];
const maxBubbles = 50;

for (let i = 0; i < maxBubbles; i++) {
  bubbles.push(createBubble());
}

Matter.Events.on(engine, 'beforeUpdate', () => {
  bubbles.forEach((bubble) => {
    const buoyancy = bubble.custom?.buoyancy || 0.0005; // 安全に
    Body.applyForce(bubble, bubble.position, { x: 0, y: -buoyancy });

    // 画面上に出たら下に戻す
    const radius = bubble.circleRadius || bubble.radius || 10;
    if (bubble.position.y + radius < 0) {
      Body.setPosition(bubble, { x: Math.random() * window.innerWidth, y: window.innerHeight + radius });
      Body.setVelocity(bubble, { x: 0, y: 0 });
    }
  });
});

window.addEventListener('resize', () => {
  render.canvas.width = window.innerWidth;
  render.canvas.height = window.innerHeight;
  render.options.width = window.innerWidth;
  render.options.height = window.innerHeight;
});
