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
    const radius = 5 + Math.random() * 15; // 大きさ
    // 大きさに応じて透明度を設定（大きいほど濃く）
    const opacity = 0.3 + (radius - 5) / 15 * 0.7;
    // 大きさに応じて浮力を設定（大きいほど速く上昇）
    const buoyancy = 0.0003 + (radius - 5) / 15 * 0.0007;

    const bubble = Bodies.circle(
      Math.random() * window.innerWidth,
      window.innerHeight + radius,
      radius,
      {
        restitution: 0.8,
        frictionAir: 0.02,
        render: {
          fillStyle: `rgba(255,255,255,${opacity})`
        },
        // buoyancy をカスタムプロパティとして保持
        custom: { buoyancy: buoyancy }
      }
    );

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
      // custom.buoyancy に応じて上方向に力を加える
      Body.applyForce(bubble, bubble.position, { x: 0, y: -bubble.custom.buoyancy });

      // 画面上に出たら下に戻す
      if (bubble.position.y + bubble.circleRadius < 0) {
        Body.setPosition(bubble, { x: Math.random() * window.innerWidth, y: window.innerHeight + bubble.circleRadius });
        Body.setVelocity(bubble, { x: 0, y: 0 });
      }
    });
  });

  window.addEventListener('resize', () => {
    render.canvas.width = window.innerWidth;
    render.canvas.height = window.innerHeight;
  });
